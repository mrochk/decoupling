import argparse
from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt
import jax, jax.numpy as jnp, optax
from torch.utils.data import DataLoader
from optax.losses import softmax_cross_entropy_with_integer_labels as celoss

from decoupling.utils import collect_information, function_error
from decoupling.scaler import JacobianScaler
from decoupling import algorithm

def forward(nn, x):
    for weights, bias in nn[:-1]:
        x = jax.nn.relu(weights @ x + bias)
    last_weights, last_bias = nn[-1]
    logits = last_weights @ x + last_bias
    return logits

def predict(nn, x): return jax.nn.softmax(forward(nn, x))

def batch_forward(nn, batch):
    return jax.vmap(partial(forward, nn))(batch)

def batch_predict(nn, batch):
    return jax.vmap(partial(predict, nn))(batch)

def load_mnist():
    from torchvision.datasets import MNIST
    from torchvision import transforms 

    transform = transforms.Compose([transforms.ToTensor(), lambda x: x.flatten().numpy()])

    train = MNIST(root='./tmp', train=True, download=True, transform=transform)
    test  = MNIST(root='./tmp', train=False, download=True, transform=transform)
    return train, test

def collate(batch):
    samples, labels = zip(*batch)
    return jnp.stack(samples) / 255.0, jnp.stack(labels)

batch_size = 128

from decoupling._common import default_dof

def kd_loss_single(teacher_logits, student_logits, temperature):
    """KL( teacher || student ) for a single example."""
    p = jax.nn.softmax(teacher_logits / temperature)   # teacher soft targets
    log_q = jax.nn.log_softmax(student_logits / temperature)  # student log-probs
    # KL(p||q) = sum p * (log p - log q)
    return jnp.sum(p * (jnp.log(p + 1e-8) - log_q))

def batch_kd_loss(teacher_logits, student_logits, temperature):
    """Mean KL loss over a batch."""
    per_example = jax.vmap(kd_loss_single, in_axes=(0, 0, None))(
        teacher_logits, student_logits, temperature
    )
    return jnp.mean(per_example)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--niters',      type=int,   required=False, default=10)
    parser.add_argument('--ntries',      type=int,   required=False, default=5)
    parser.add_argument('--temperature', type=float, required=False, default=4.0,
                        help='Distillation temperature T (default: 4.0)')
    args = parser.parse_args()

    niters      = args.niters
    ntries      = args.ntries
    temperature = args.temperature

    key = jax.random.key(0)

    train, test = load_mnist()

    trainloader = DataLoader(train, batch_size=batch_size, collate_fn=collate)
    testloader  = DataLoader(test,  batch_size=batch_size, collate_fn=collate)

    params = jnp.load('./tmp/weights.npz')
    params = [(params['W1'], params['b1']), (params['W2'], params['b2'])]

    X = next(iter(trainloader))[0]
    N = X.shape[0]

    print(X.shape)
    print(default_dof(N))

    X, Y, J = collect_information(lambda x: forward(params, x), N, key, 784, X=X)

    scaler = JacobianScaler(J)
    J_scaled, Y_scaled = scaler.scale(J, Y)
    scaling_factors = scaler.factors

    def dforward(dparams, x):
        V, W = dparams
        return (W @ decoupling.internals(V.T @ x)) / scaling_factors

    for rank in [64]:

        print(f'RANK = {rank}')

        min_err = jnp.inf
        best_decoupling = None

        def batch_dforward(dparams, X):
            return jax.vmap(partial(dforward, dparams))(X)

        def evaluate_decoupling(dparams, loader):
            accuracy = 0.0
            for i, (X, y) in enumerate(tqdm(loader, desc='Decoupling Evaluation')):
                logits = batch_dforward(dparams, X)
                preds = jnp.argmax(jax.nn.softmax(logits), -1)
                accuracy += (jnp.mean(preds == y) - accuracy) / (i+1)
            print(f'\nAccuracy = {accuracy*100:.2f}% ===================\n')

        for k in jax.random.split(key, 3):

            algo = algorithm.CMTF_PSpline(rank, key=k, niters=50)
            decoupling = algo.run(X, Y_scaled, J_scaled)

            dparams = [decoupling.V, decoupling.W]

            evaluate_decoupling(dparams, testloader)

            errs = function_error(lambda x: forward(params, x), lambda x: dforward(dparams, x), X)
            print(jnp.mean(errs))

            if jnp.mean(errs) < min_err:
                best_decoupling = decoupling
                min_err = jnp.mean(errs)

        decoupling = best_decoupling
        (W, V, _, _) = decoupling.factors

        Z = X @ V
        O = jax.vmap(decoupling.internals)(Z)

        ncols = min(4, rank)
        nrows = (rank + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))
        axes = axes.flatten()

        for r in range(rank):
            z = Z[:, r]
            y = O[:, r]
            idx = jnp.argsort(z)
            zs = z[idx]
            ys = y[idx]
            axes[r].plot(zs, ys, color='black', linewidth=3)
            axes[r].set_yticks([])
            axes[r].set_xticks([])

        plt.tight_layout()
        plt.savefig(f'internals.png', dpi=300)
        plt.close(fig)

        dparams = [V, W]

        def batch_dforward(dparams, X):
            return jax.vmap(partial(dforward, dparams))(X)

        preds_decoupling = jnp.argmax(batch_dforward(dparams, X), axis=-1)
        preds_nn = jnp.argmax(batch_predict(params, X), axis=-1)
        same = jnp.mean(preds_decoupling == preds_nn)
        print(f'Same = {same*100:.2f}%')

        def evaluate_decoupling(dparams, loader):
            accuracy = 0.0
            for i, (X, y) in enumerate(tqdm(loader, desc='Decoupling Evaluation')):
                logits = batch_dforward(dparams, X)
                preds = jnp.argmax(jax.nn.softmax(logits), -1)
                accuracy += (jnp.mean(preds == y) - accuracy) / (i+1)
            print(f'\nAccuracy = {accuracy*100:.2f}% ===================\n')

        def ce_loss(dparams, X, y):
            logits = batch_dforward(dparams, X)
            return jnp.mean(celoss(logits, y))

        def dcompute_kd_loss(dparams, X, y, temperature):
            W, V = dparams
            V = jax.lax.stop_gradient(V)
            student_logits = batch_dforward([W, V], X)
            # Teacher logits: stop gradient so we don't backprop into params
            teacher_logits = jax.lax.stop_gradient(batch_forward(params, X))
            return ce_loss(dparams, X, y) + 0.5*batch_kd_loss(teacher_logits, student_logits, temperature)

        doptim = optax.adam(learning_rate=1e-3)
        dstate = doptim.init(dparams)

        def dstep(dparams, dstate, X, y, temperature):
            loss, grads = jax.value_and_grad(dcompute_kd_loss)(dparams, X, y, temperature)
            updates, dstate = doptim.update(grads, dstate)
            dparams = optax.apply_updates(dparams, updates)
            return dparams, dstate, loss

        # Evaluate before fine-tuning
        evaluate_decoupling(dparams, testloader)

        bar = tqdm(trainloader, desc=f'Fine-tuning (KD, T={temperature})')
        for _, (Xb, yb) in enumerate(bar):
            # Note: true labels yb are not used — pure distillation
            dparams, dstate, loss = dstep(dparams, dstate, Xb, yb, temperature)
            bar.set_postfix(loss=f'{loss:.4f}')

        # Evaluate after fine-tuning
        evaluate_decoupling(dparams, testloader)

        preds_decoupling = jnp.argmax(batch_dforward(dparams, X), axis=-1)
        preds_nn = jnp.argmax(batch_predict(params, X), axis=-1)
        same = jnp.mean(preds_decoupling == preds_nn)
        print(f'Same = {same*100:.2f}%')

if __name__ == '__main__': main()
