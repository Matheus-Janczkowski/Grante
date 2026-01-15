# Routine to assemble the whole residual vector


@tf.function
def total_energy(F):
    return tf.add_n(
        mask_k * energy_k(F, fields_k)
        for k in materials
    )

@tf.function
def residual(F):
    with tf.GradientTape() as tape:
        tape.watch(F)
        psi = total_energy(F)
    return tape.gradient(psi, F)

for it in range(max_iter):
    F = compute_F(u)
    R = residual(F)
    u = update(u, R)