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


def test_matrix_transpose():
    mat = Matrix(init=[
        [1, 2, 3],
        [4, 5, 6]
    ])

    result = mat.transpose

    print(result.array)

    assert result.array == [
        [1, 4],
        [2, 5],
        [3, 6],
    ]


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

    print(result.array)

    assert result.array == [
        [3, 5, 7],
        [7, 7, 7]
    ]


def test_matrix_sub():
    mat1 = Matrix(init=[
        [1, 2, 3],
        [4, 5, 6]
    ])

    mat2 = Matrix(init=[
        [2, 3, 4],
        [3, 2, 1],
    ])

    result = mat1 - mat2

    print(result.array)

    assert result.array == [
        [-1, -1, -1],
        [1, 3, 5]
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


def test_matrix_matmul_matrix():
    mat1 = Matrix(init=[
        [1, 2, 3],
        [4, 5, 6]
    ])

    result = mat1 @ mat1

    print(result.array)

    assert result.array == [
        [1, 4, 9],
        [16, 25, 36],
    ]


def test_matrix_matmul_number():
    mat1 = Matrix(init=[
        [1, 2, 3],
        [4, 5, 6]
    ])

    result1 = mat1 @ 2
    result2 = mat1 @ 2.0

    print(result1.array)
    print(result2.array)

    assert result1.array == [
        [2, 4, 6],
        [8, 10, 12],
    ]
    assert result1.array == [
        [2.0, 4.0, 6.0],
        [8.0, 10.0, 12.0],
    ]


def test_matrix_neg():
    mat = Matrix(init=[
        [1, 2, 3],
        [4, 5, 6],
    ])

    result = -mat

    print(result.array)
    assert result.array == [
        [-1, -2, -3],
        [-4, -5, -6],
    ]
