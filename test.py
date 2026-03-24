if __name__ == "__main__":
    import torch
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
    from train import TrainingResult
    from plot_utilities import set_font_size

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

    test_set = SynteticDataset(
        ground_truth=BSDS500(
            path="data/BSDS500/test",
            transform=transforms.CenterCrop(size),
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

    model.load_state_dict(torch.load(save_path))
    model.to(device)

    create_directory(save_path)
    create_directory(info_save_path)

    ssim_fun = SSIM(data_range=1.0, size_average=True, channel=1)

    import json

    with open(results_save_path, "r") as f:
        results = json.load(f)
        results = TrainingResult(**results)

    import numpy as np
    from matplotlib import pyplot as plt

    execution_time = np.cumsum(results.execution_time)

    plots_folder = "/plots/"
    create_directory(save_folder + save_name + plots_folder)

    # loss is -SSIM, so we negate it for plotting
    results.loss_history = -np.array(results.loss_history)
    results.val_loss_history = -np.array(results.val_loss_history)

    set_font_size(18)
    plt.figure()
    plt.semilogy(execution_time, results.loss_history, label="Training loss")
    plt.semilogy(execution_time, results.val_loss_history, label="Validation loss")
    plt.xlabel("Execution time (s)")
    plt.ylabel("$-\mathcal{L}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_folder + save_name + plots_folder + "loss_over_time.pdf")

    plt.figure()
    epochs = range(len(results.loss_history))
    plt.semilogy(epochs, results.loss_history, label="Training loss")
    plt.semilogy(epochs, results.val_loss_history, label="Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("$-\mathcal{L}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_folder + save_name + plots_folder + "loss_over_epochs.pdf")

    i_list = [0, 3, 6, 8]
    for i in i_list:
        observed, gt = test_set[i]
        observed_device = observed.unsqueeze(0).to(device)
        gt_device = gt.unsqueeze(0).to(device)

        square_fig_size = 6
        square_fig_size = (square_fig_size, square_fig_size)

        set_font_size(22)
        plt.figure(figsize=square_fig_size)
        plt.imshow(gt.cpu().squeeze(), cmap="gray")
        plt.axis("off")
        plt.title(" ")
        plt.tight_layout()
        plt.savefig(save_folder + save_name + plots_folder + f"ground_truth_{i}.pdf")

        plt.figure(figsize=square_fig_size)
        plt.imshow(observed.cpu().squeeze(), cmap="gray")
        plt.title(f"SSIM: {ssim_fun(observed_device, gt_device).item():.4f}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_folder + save_name + plots_folder + f"observed_{i}.pdf")

        alpha_list = []
        beta_list = []
        lam_list = []
        nu_list = []
        ssim_list = []

        with torch.no_grad():
            u = observed_device.clone()
            v = torch.zeros((2, *u.shape), device=device)
            ssim_list.append(ssim_fun(u, gt_device).item())

            model.eval()
            for _ in range(100):
                u, v = model(observed_device, u, v)
                alpha_list.append(model.last_alpha.item())
                beta_list.append(model.last_beta.item())
                lam_list.append(model.last_lam.item())
                nu_list.append(model.last_nu.item())
                ssim_list.append(ssim_fun(u, gt_device).item())

            plt.figure(figsize=square_fig_size)
            plt.imshow(u.cpu().squeeze(), cmap="gray")
            plt.axis("off")
            plt.title(f"SSIM: {ssim_fun(u, gt_device).item():.4f}")
            plt.tight_layout()
            plt.savefig(
                save_folder + save_name + plots_folder + f"reconstruction_{i}.pdf"
            )

        set_font_size(28)
        plt.figure()
        plt.plot(alpha_list, label=r"$\alpha$")
        plt.plot(beta_list, label=r"$\beta$")
        # plt.xlabel("Iterations")
        # plt.ylabel("Step size")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            save_folder
            + save_name
            + plots_folder
            + f"step_sizes_over_iterations_{i}.pdf"
        )

        plt.figure()
        plt.plot(lam_list, label=r"$\lambda$")
        # plt.xlabel("Iterations")
        # plt.ylabel(r"$\lambda$")
        plt.tight_layout()
        plt.savefig(
            save_folder
            + save_name
            + plots_folder
            + f"regularization_parameter_over_iterations_{i}.pdf"
        )

        plt.figure()
        plt.plot(nu_list, label=r"$\nu$")
        # plt.xlabel("Iterations")
        # plt.ylabel(r"$\nu$")
        plt.tight_layout()
        plt.savefig(
            save_folder
            + save_name
            + plots_folder
            + f"preconditioner_parameter_over_iterations_{i}.pdf"
        )

        plt.figure()
        plt.plot(ssim_list, label="SSIM")
        # plt.xlabel("Iterations")
        # plt.ylabel("SSIM")
        plt.tight_layout()
        plt.savefig(
            save_folder + save_name + plots_folder + f"ssim_over_iterations_{i}.pdf"
        )

    plt.show()
