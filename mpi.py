import numpy as np


def binarizar(imagen_gris, umbral=127):
    """Convierte una imagen en escala de grises a binaria invertida."""
    return np.where(imagen_gris > umbral, 0, 255).astype(np.uint8)


def dilatar(imagen_binaria, kernel):
    """Aplica dilatación binaria usando únicamente NumPy."""
    height, width = imagen_binaria.shape
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    padded = np.pad(
        imagen_binaria,
        ((pad_height, pad_height), (pad_width, pad_width)),
        mode="constant",
        constant_values=0,
    )
    output = np.zeros_like(imagen_binaria)

    for y in range(height):
        for x in range(width):
            region = padded[y : y + kernel_height, x : x + kernel_width]
            output[y, x] = 255 if np.any(region & kernel * 255) else 0

    return output


def erosionar(imagen_binaria, kernel):
    """Aplica erosión binaria usando únicamente NumPy."""
    height, width = imagen_binaria.shape
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    padded = np.pad(
        imagen_binaria,
        ((pad_height, pad_height), (pad_width, pad_width)),
        mode="constant",
        constant_values=0,
    )
    output = np.zeros_like(imagen_binaria)

    for y in range(height):
        for x in range(width):
            region = padded[y : y + kernel_height, x : x + kernel_width]
            output[y, x] = 255 if np.all(region & kernel * 255) else 0

    return output


def cerrar(imagen_binaria, kernel_size=5, iteraciones=2):
    """Aplica una operación de cierre morfológico."""
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    output = imagen_binaria.copy()

    for _ in range(iteraciones):
        output = dilatar(output, kernel)
    for _ in range(iteraciones):
        output = erosionar(output, kernel)

    return output


def tiene_pastilla(
    parche_gray,
    umbral=127,
    kernel_size=5,
    iteraciones=2,
    umbral_negro=0.2,
):
    """Determina si el parche analizado parece contener una pastilla."""
    parche_bin = binarizar(parche_gray, umbral)
    parche_cerrado = cerrar(parche_bin, kernel_size, iteraciones)

    height, width = parche_cerrado.shape
    margin = int(min(height, width) * 0.2)
    center = parche_cerrado[
        margin : height - margin,
        margin : width - margin,
    ]

    if center.size == 0:
        return False

    black_ratio = np.sum(center == 255) / center.size
    return black_ratio < umbral_negro
