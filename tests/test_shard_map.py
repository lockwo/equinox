import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.sharding as jshard
import pytest


pytestmark = pytest.mark.skipif(
    jax.default_backend() != "cpu",
    reason=(
        "Shard map tests require multiple devices. JAX can simulate multiple "
        "devices on a single CPU but cannot easily do so on a single GPU/TPU. "
        "Therefore, we skip these tests on non-CPU backends."
    ),
)

P = jshard.PartitionSpec


def _make_mesh():
    num_devices = len(jax.devices())
    assert num_devices >= 2
    return jax.make_mesh((num_devices,), ("batch",))


def test_basic_sharding():
    mesh = _make_mesh()
    num_devices = len(jax.devices())

    @eqx.filter_jit
    @eqx.filter_shard_map(mesh=mesh, in_specs=(P(), P("batch")), out_specs=P("batch"))
    def f(w, x):
        return x * w

    w = jnp.array(2.0)
    x = jnp.arange(num_devices * 4, dtype=jnp.float32)
    assert jnp.allclose(f(w, x), x * 2.0)

    g = eqx.filter_jit(
        eqx.filter_shard_map(
            lambda w, x: x + w,
            mesh=mesh,
            in_specs=(P(), P("batch")),
            out_specs=P("batch"),
        )
    )
    assert jnp.allclose(g(w, x), x + 2.0)


def test_module_input():
    mesh = _make_mesh()
    num_devices = len(jax.devices())

    mlp = eqx.nn.MLP(3, 3, 64, 2, key=jr.key(0))
    x = jnp.ones((num_devices * 2, 3))

    @eqx.filter_jit
    @eqx.filter_shard_map(mesh=mesh, in_specs=(P(), P("batch")), out_specs=P("batch"))
    def f(model, x):
        return jax.vmap(model)(x)

    out = f(mlp, x)
    expected = jax.vmap(mlp)(x)
    assert jnp.allclose(out, expected)


def test_non_array_output():
    mesh = _make_mesh()
    num_devices = len(jax.devices())

    @eqx.filter_jit
    @eqx.filter_shard_map(mesh=mesh, in_specs=P("batch"), out_specs=(P("batch"), P()))
    def f(x):
        return x + 1.0, "metadata"

    x = jnp.ones(num_devices * 2)
    arr, meta = f(x)
    assert jnp.allclose(arr, x + 1.0)
    assert meta == "metadata"


def test_collectives():
    mesh = _make_mesh()
    num_devices = len(jax.devices())

    @eqx.filter_jit
    @eqx.filter_shard_map(mesh=mesh, in_specs=P("batch"), out_specs=P())
    def f(x):
        return lax.psum(jnp.sum(x), "batch")

    x = jnp.ones(num_devices * 4)
    assert jnp.allclose(f(x), jnp.array(num_devices * 4.0))


def test_grad_inside_shard_map():
    mesh = _make_mesh()
    num_devices = len(jax.devices())

    mlp = eqx.nn.MLP(2, 2, 16, 1, key=jr.key(0))
    x = jnp.ones((num_devices * 2, 2))

    @eqx.filter_jit
    @eqx.filter_shard_map(mesh=mesh, in_specs=(P(), P("batch")), out_specs=(P(), P()))
    def loss_and_grad(model, x):
        @eqx.filter_value_and_grad
        def loss_fn(model, x):
            pred = jax.vmap(model)(x)
            return jnp.mean(pred**2)

        loss, grads = loss_fn(model, x)
        loss = lax.pmean(loss, "batch")
        grads = lax.pmean(grads, "batch")
        return loss, grads

    loss, grads = loss_and_grad(mlp, x)
    assert loss.shape == ()
    assert not jnp.isnan(loss)
    grad_leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
    assert len(grad_leaves) > 0
    assert all(not jnp.any(jnp.isnan(g)) for g in grad_leaves)
