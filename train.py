import torch
from torch.utils.data import DataLoader
import time
from dataclasses import dataclass
from typing import Callable


@dataclass
class TrainingConfig:
    model: torch.nn.Module
    training_set: torch.utils.data.Dataset
    validation_set: torch.utils.data.Dataset
    batch_size: int
    unrolling_steps: int = 18  # number of unrolling steps (default 1 for no unrolling)
    loss_fn: Callable = torch.nn.MSELoss()
    optimizer: Callable = lambda param: torch.optim.Adam(param, lr=0.001)
    epochs: int = None  # number of epochs (None for infinite)
    max_time: float = None  # maximum time in seconds (None for infinite)
    save_path: str = None  # path to save the model
    info_path: str = None  # path to save the training info
    num_threads: int = 8  # number of threads (0 for all available)
    device: str = "cpu"


@dataclass
class TrainingResult:
    execution_time: list[float]
    loss_history: list[float]
    val_loss_history: list[float]
    device: str
    torch_config: str
    torch_num_threads: int
    torch_seed: int


def training(config: TrainingConfig):
    device = config.device
    if device == "cuda":
        torch.multiprocessing.set_start_method("spawn")

    # Set the number of threads
    if config.num_threads > 0:
        torch.set_num_threads(config.num_threads)

    print(f"Using device {device}")
    print("torch config:\n", torch.__config__.show())
    print("torch threading:\n", torch.get_num_threads())

    # Create the data loaders
    num_workers = 0
    train_loader = DataLoader(
        config.training_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    validation_loader = DataLoader(
        config.validation_set,
        batch_size=len(config.validation_set),
        shuffle=False,
        num_workers=num_workers,
    )

    # Set up the model, loss function, and optimizer
    model = config.model
    if device == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)
    loss_fn = config.loss_fn
    optimizer = config.optimizer(model.parameters())

    best_loss = float("inf")
    execution_time = []
    loss_history = []
    val_loss_history = []
    elapsed_time = 0.0

    # Train the model
    epoch = -1

    if config.epochs is not None:
        epoch_condition = lambda epoch: epoch < config.epochs  # noqa: E731
    else:
        epoch_condition = lambda epoch: True  # noqa: E731
    if config.max_time is not None:
        time_condition = lambda elapsed_time: elapsed_time < config.max_time  # noqa: E731
    else:
        time_condition = lambda elapsed_time: True  # noqa: E731
    should_continue = lambda epoch, elapsed_time: (  # noqa: E731
        epoch_condition(epoch) and time_condition(elapsed_time)
    )

    while should_continue(epoch, elapsed_time):
        epoch += 1
        running_loss = 0.0
        last_loss = 0.0
        val_running_loss = 0.0

        t0 = time.perf_counter()
        model.train()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()

            observed, ground_truth = data
            observed = observed.to(device)
            ground_truth = ground_truth.to(device)
            u = observed.clone()
            v = torch.zeros((2, *observed.shape), device=device)

            loss = torch.tensor(0.0, device=device)
            coeff_sum = 0.0

            # Make predictions for this batch
            for i in range(config.unrolling_steps):
                u, v = model(observed, u, v)
                # coeff = 1.
                coeff = 1.0 / (config.unrolling_steps - i)
                coeff_sum += coeff
                loss += loss_fn(u, ground_truth) * coeff
            loss = loss / coeff_sum

            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
        epoch_time = time.perf_counter() - t0
        execution_time.append(epoch_time)
        elapsed_time += epoch_time

        last_loss = running_loss / (i + 1)
        loss_history.append(last_loss)
        running_loss = 0.0

        model.eval()
        with torch.no_grad():
            for val_i, val_data in enumerate(validation_loader):
                val_observed, val_ground_truth = val_data
                val_observed = val_observed.to(device)
                val_ground_truth = val_ground_truth.to(device)
                val_u = val_observed.clone()
                val_v = torch.zeros((2, *val_observed.shape), device=device)

                val_loss = torch.tensor(0.0, device=device)

                for i in range(config.unrolling_steps):
                    val_u, val_v = model(val_observed, val_u, val_v)
                    val_loss += loss_fn(val_u, val_ground_truth)
                val_loss = val_loss / config.unrolling_steps
                val_running_loss += val_loss.item()

        val_last_loss = val_running_loss / (val_i + 1)
        val_loss_history.append(val_last_loss)
        val_running_loss = 0.0

        info = f"epoch {epoch}"
        minutes, seconds = divmod(elapsed_time, 60)
        info += f" elapsed time: {int(minutes)}:{int(seconds):02d}"
        info += f" train loss: {last_loss}"
        info += f" validation loss: {val_last_loss}"

        if val_last_loss < best_loss:
            best_loss = val_last_loss
            if config.save_path is not None:
                torch.save(model.state_dict(), config.save_path)
                print(f"New best model saved with loss {best_loss}")
            if config.info_path is not None:
                with open(config.info_path, "w") as f:
                    f.write(f"{info}\n")

        print(info)

    results = TrainingResult(
        execution_time=execution_time,
        loss_history=loss_history,
        val_loss_history=val_loss_history,
        device=str(device),
        torch_config=torch.__config__.show(),
        torch_num_threads=torch.get_num_threads(),
        torch_seed=torch.random.initial_seed(),
    )
    return results


if __name__ == "__main__":
    from datasets.SynteticDataset import SynteticDataset
    from models.ConvNet import ConvNet as Net
    from models.PNPD_NN_step import PNPD_NN_step
    from utilities import create_directory
    from datasets.BSDS500 import BSDS500
    from torchvision import transforms
    from torch.fft import rfft2
    from math_extras import (
        blur_image,
        gaussian_psf,
        grad2D,
        div2D,
        grad_least_squares_blur,
        P_inv,
        prox_h_star_TV,
    )
    from solvers import PNPD_functions, PNPD_parameters
    from pytorch_msssim import SSIM

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    size = (256, 256)

    psf_rfft2 = rfft2(gaussian_psf(sigma=2, size=size)).to(device)
    psf_rfft2_conj = psf_rfft2.conj().to(device)
    psf_rfft2_abs_sq = psf_rfft2 * psf_rfft2_conj

    def blur_operator(image):
        return blur_image(image, psf_rfft2)

    noise_percent = 0.01  # 1% noise

    def noise_operator(image):
        noise = torch.randn_like(image)
        noise /= noise.norm()
        noise *= noise_percent * image.norm()
        return image + noise

    def measurement_operator(image):
        return noise_operator(blur_operator(image))

    training_set = SynteticDataset(
        ground_truth=BSDS500(
            path="data/BSDS500/train",
            transform=transforms.RandomCrop(size),
            device=device,
        ),
        transform=measurement_operator,
    )
    validation_set = SynteticDataset(
        ground_truth=BSDS500(
            path="data/BSDS500/val",
            transform=transforms.RandomCrop(size),
            device=device,
        ),
        transform=measurement_operator,
    )

    functions = PNPD_functions(
        P_inv=None,  # Will be set in the model
        prox_h_star=None,  # Will be set in the model
        grad_f=None,  # Will be set in the model
        W=grad2D,
        W_T=div2D,
    )

    parameters = PNPD_parameters(
        alpha=None,  # Will be set in the model
        beta=None,  # Will be set in the model
        k_max=1,
    )

    model = PNPD_NN_step(
        alpha_Net=Net(),
        beta_Net=Net(),
        nu_Net=Net(),
        lam_Net=Net(),
        params=parameters,
        funs=functions,
        P_inv=lambda nu, x: P_inv(
            x=x, nu=nu, psfAbsSq=psf_rfft2_abs_sq
        ),  # nu will be set in the model
        prox_h_star=lambda lambda_regularization, v, lambda_prox: prox_h_star_TV(
            lambda_regularization, v
        ),  # lambda_regularization will be set in the model
        grad_f=lambda u, observed_rfft: grad_least_squares_blur(
            u, observed_rfft, psf_rfft2, psf_rfft2_conj=psf_rfft2_conj
        ),  # observed_rfft will be set in the model
    )

    save_folder = "results/"
    save_name = "PNPD_unrolling_SSIM"
    save_path = save_folder + save_name + ".pth"
    info_save_path = save_folder + save_name + "_info.txt"
    results_save_path = save_folder + save_name + "_results.json"

    create_directory(save_path)
    create_directory(info_save_path)

    ssim_fun = SSIM(data_range=1.0, size_average=True, channel=1)

    config = TrainingConfig(
        model=model,
        training_set=training_set,
        validation_set=validation_set,
        batch_size=8,
        save_path=save_path,
        info_path=info_save_path,
        device=device,
        optimizer=lambda param: torch.optim.Adam(param, lr=1e-5),
        loss_fn=lambda X, Y: -ssim_fun(X, Y),  # Using SSIM as loss function
        epochs=50,
    )

    results = training(config)

    import json
    from dataclasses import asdict

    with open(results_save_path, "w") as f:
        json.dump(asdict(results), f)

    import numpy as np
    from matplotlib import pyplot as plt

    execution_time = np.cumsum(results.execution_time)

    plots_folder = "/plots/"
    create_directory(save_folder + save_name + plots_folder)

    plt.figure()
    plt.semilogy(execution_time, results.loss_history, label="Training loss")
    plt.semilogy(execution_time, results.val_loss_history, label="Validation loss")
    plt.xlabel("Execution time (s)")
    plt.ylabel("Loss")
    plt.title("Loss over time")
    plt.legend()
    plt.savefig(save_folder + save_name + plots_folder + "loss_over_time.png")

    plt.figure()
    epochs = range(len(results.loss_history))
    plt.semilogy(epochs, results.loss_history, label="Training loss")
    plt.semilogy(epochs, results.val_loss_history, label="Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss over epochs")
    plt.legend()
    plt.savefig(save_folder + save_name + plots_folder + "loss_over_epochs.png")

    f, axs = plt.subplots(1, 3)
    i = 1
    ax1, ax2, ax3 = axs
    observed, gt = validation_set[i]

    ax1.imshow(gt.cpu().squeeze(), cmap="gray")
    ax1.axis("off")
    ax1.set_title("Ground Truth")

    ax2.imshow(observed.cpu().squeeze(), cmap="gray")
    ax2.axis("off")
    ax2.set_title("Observed")

    alpha_list = []
    beta_list = []
    lam_list = []
    nu_list = []
    ssim_list = []

    with torch.no_grad():
        u = observed.clone().unsqueeze(0).to(device)
        v = torch.zeros((2, *u.shape), device=device)
        gt_device = gt.unsqueeze(0).to(device)
        ssim_list.append(ssim_fun(u, gt_device).item())

        model.eval()
        for _ in range(100):
            u, v = model(observed.unsqueeze(0).to(device), u, v)
            alpha_list.append(model.last_alpha.item())
            beta_list.append(model.last_beta.item())
            lam_list.append(model.last_lam.item())
            nu_list.append(model.last_nu.item())
            ssim_list.append(ssim_fun(u, gt_device).item())

        ax3.imshow(u.cpu().squeeze(), cmap="gray")
        ax3.axis("off")
        ax3.set_title("Reconstructed")

    plt.savefig(save_folder + save_name + plots_folder + "reconstruction_example.png")

    plt.figure()
    plt.plot(alpha_list, label=r"$\alpha$")
    plt.plot(beta_list, label=r"$\beta$")
    plt.xlabel("Iterations")
    plt.ylabel("Step size")
    plt.title("Step sizes over iterations")
    plt.legend()
    plt.savefig(
        save_folder + save_name + plots_folder + "step_sizes_over_iterations.png"
    )

    plt.figure()
    plt.plot(lam_list, label=r"$\lambda$")
    plt.xlabel("Iterations")
    plt.ylabel(r"$\lambda$")
    plt.title("Regularization parameter over iterations")
    plt.savefig(
        save_folder
        + save_name
        + plots_folder
        + "regularization_parameter_over_iterations.png"
    )

    plt.figure()
    plt.plot(nu_list, label=r"$\nu$")
    plt.xlabel("Iterations")
    plt.ylabel(r"$\nu$")
    plt.title("Preconditioner parameter over iterations")
    plt.savefig(
        save_folder
        + save_name
        + plots_folder
        + "preconditioner_parameter_over_iterations.png"
    )

    plt.figure()
    plt.plot(ssim_list, label="SSIM")
    plt.xlabel("Iterations")
    plt.ylabel("SSIM")
    plt.title("SSIM over iterations")
    plt.savefig(save_folder + save_name + plots_folder + "ssim_over_iterations.png")

    plt.show()
