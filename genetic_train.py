import torch
from train import training, TrainingConfig


def fitness(model, validation_loader, device, fitness_fn, unrolling_steps):
    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for val_i, val_data in enumerate(validation_loader):
            val_observed, val_ground_truth = val_data
            val_observed = val_observed.to(device)
            val_ground_truth = val_ground_truth.to(device)
            val_u = val_observed.clone()
            val_v = torch.zeros((2, *val_observed.shape), device=device)

            val_loss = torch.tensor(0.0, device=device)

            for i in range(unrolling_steps):
                val_u, val_v = model(val_observed, val_u, val_v)
                val_loss += fitness_fn(val_u, val_ground_truth)
            val_loss = val_loss / unrolling_steps
            val_running_loss += val_loss.item()

    val_last_loss = val_running_loss / (val_i + 1)
    return val_last_loss


def generate_child(parent1, parent2, model_factory):
    child = model_factory().to(device)
    for child_param, param1, param2 in zip(
        child.parameters(), parent1.parameters(), parent2.parameters()
    ):
        if torch.rand(1).item() < 0.5:
            child_param.data.copy_(param1.data)
        else:
            child_param.data.copy_(param2.data)
    return child


def mutate(model, mutation_rate=0.01):
    for param in model.parameters():
        if torch.rand(1).item() < mutation_rate:
            noise = torch.randn_like(param) * 0.1  # Small noise
            param.data += noise
    return model


def next_generation(population, fitness_scores, population_size, model_factory):
    probabilities = torch.softmax(fitness_scores, dim=0)
    parents_1 = torch.multinomial(probabilities, population_size, replacement=True)
    parents_2 = torch.multinomial(probabilities, population_size, replacement=True)
    parents_1 = [population[i] for i in parents_1]
    parents_2 = [population[i] for i in parents_2]
    parents_pairs = zip(parents_1, parents_2)
    children = [
        generate_child(parent1, parent2, model_factory)
        for parent1, parent2 in parents_pairs
    ]
    children = [mutate(child) for child in children]
    return children


def genetic_train(
    population_size,
    model_factory,
    training_loader,
    fitness_fn,
    unrolling_steps,
    n_best_models=5,
    device="cpu",
    grad_steps=1,
):
    population = [model_factory().to(device).to(device) for _ in range(population_size)]
    fitness_scores = torch.zeros(population_size)

    for generation in range(100):  # Example: 100 generations
        for i, model in enumerate(population):
            fitness_scores[i] = fitness(
                model,
                training_loader,
                device=device,
                fitness_fn=fitness_fn,
                unrolling_steps=unrolling_steps,
            )
            while torch.isnan(fitness_scores[i]):
                print(f"Model {i} produced NaN loss, regenerating.")
                population[i] = model_factory().to(device)
                fitness_scores[i] = fitness(
                    population[i],
                    training_loader,
                    device=device,
                    fitness_fn=fitness_fn,
                    unrolling_steps=unrolling_steps,
                )

            config = TrainingConfig(
                model=model,
                training_set=training_set,
                validation_set=training_set,
                unrolling_steps=10,
                batch_size=2,
                device=device,
                optimizer=lambda param: torch.optim.Adam(param, lr=1e-5),
                loss_fn=lambda X, Y: -ssim_fun(X, Y),  # Using SSIM as loss function
                epochs=grad_steps - 1,
            )

            training(config)

            fitness_scores[i] = fitness(
                population[i],
                training_loader,
                device=device,
                fitness_fn=fitness_fn,
                unrolling_steps=unrolling_steps,
            )
            if torch.isnan(fitness_scores[i]):
                fitness_scores[
                    i
                ] = -100  # Assign a large negative value to NaN fitness scores

            print(f"Model {i}, Fitness: {fitness_scores[i]}")

        best_fitness_scores, indices = torch.topk(fitness_scores, n_best_models)
        best_models = [population[idx] for idx in indices]

        info = f"Generation: {generation}"
        info += f", Best fitness: {best_fitness_scores[0].item()}"
        print(info)

        population = next_generation(
            population, fitness_scores, population_size, model_factory
        )

    return best_models


if __name__ == "__main__":
    from datasets.SynteticDataset import SynteticDataset
    from models.ConvNet import ConvNet as Net
    from models.PNPD_NN_step import PNPD_NN_step
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
    from torch.utils.data import DataLoader

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
    # validation_set = SynteticDataset(
    #   ground_truth=BSDS500(path="data/BSDS500/val", transform=transforms.RandomCrop(size), device=device),
    #   transform=measurement_operator
    # )

    num_workers = 0
    train_loader = DataLoader(
        training_set,
        batch_size=len(training_set),
        shuffle=False,
        num_workers=num_workers,
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

    def model_factory():
        return PNPD_NN_step(
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

    ssim_fun = SSIM(data_range=1.0, size_average=True, channel=1)

    best_models = genetic_train(
        population_size=4,
        model_factory=model_factory,
        training_loader=train_loader,
        fitness_fn=lambda X, Y: -ssim_fun(X, Y),  # Using SSIM as fitness function
        unrolling_steps=20,
        n_best_models=1,
        device=device,
    )
