from nn.matrix import Matrix


def test_matrix_init():
    mat1 = Matrix(1, 2)
    assert mat1.xsize == 2
    assert mat1.ysize == 1

    init_array = [[1, 2, 3], [4, 5, 6]]
    mat2 = Matrix(init=init_array)
    assert mat2.xsize == 3
    assert mat2.ysize == 2
    assert mat2.array == init_array


def test_matrix_add():
    mat1 = Matrix(init=[
        [1, 2, 3],
        [4, 5, 6]
    ])

    mat2 = Matrix(init=[
        [2, 3, 4],
        [3, 2, 1],
    ])

    result = mat1 + mat2

    assert result.array == [
        [3, 5, 7],
        [7, 7, 7]
    ]

def test_matrix_mul():
    mat1 = Matrix(init=[
        [1, 2, 3],
        [4, 5, 6]
    ])

    mat2 = Matrix(init=[
        [1],
        [2],
        [3]
    ])

    result = mat1 * mat2

    print(mat1.array)
    print(mat2.array)
    print(result.array)

    assert result.array == [[14], [32]]
