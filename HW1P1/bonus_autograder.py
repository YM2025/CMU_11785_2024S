import mytorch.nn.dropout

from mytorch.models.mlp import MLP4
from mytorch.optim.adam import Adam
from mytorch.optim.adamW import AdamW

import numpy as np
np.set_printoptions(precision=4)

import json
import pickle

def reset_prng():
    np.random.seed(11785)

base_dir = "."

def print_status_message(test_number, check_result, description):
    
    FAIL_MESSAGE  = " │\033[31m FAILED \033[0m│ "
    PASS_MESSAGE  = " │\033[32m PASSED \033[0m│ "
    status_string = PASS_MESSAGE if check_result else FAIL_MESSAGE
    
    print("\n")
    print("TEST         | STATUS | DESCRIPTION")
    print("─────────────┼────────┼─────────────────────────────────────────")
    print("Test " + test_number, status_string, description, sep="")
    print("\n")
    
    return None

def print_header(string):
    
    START_HEADER = "\033[1;37;40m"
    END_HEADER   = "\033[0m"
    
    print(START_HEADER + "{:<64}".format(string) + END_HEADER)
    
    return None

def print_array_details(x, digits=4):
    
    assert type(x) is np.ndarray, str(type(x))+"is not <class 'numpy.ndarray'>"
    
    BOLD_START = "\033[1m"
    BOLD_END   = "\033[0m"
    
    print(BOLD_START, "\nDTYPE:\n", BOLD_END, x.dtype, sep="")
    print(BOLD_START, "\nSHAPE:\n", BOLD_END, x.shape, sep="")
    print(BOLD_START, "\nVALUE:\n", BOLD_END, np.round(x, digits), sep="")
    print()
    
    return None

def print_scalar_details(x, digits=4):
        
    BOLD_START = "\033[1m"
    BOLD_END   = "\033[0m"
    
    print(BOLD_START, "\nDTYPE:\n", BOLD_END, type(x), sep="")
    print(BOLD_START, "\nVALUE:\n", BOLD_END, np.round(x, digits), sep="")
    print()
    
    return None

def MLP4_forward(mlp4):

    A0 = np.array([
        [-4., -3.],
        [-2., -1.],
        [ 0.,  1.],
        [ 2.,  3.],
        [ 4.,  5.]], dtype="f")

    W0 = np.array([
        [0., 1.],
        [1., 2.],
        [2., 0.],
        [0., 1.]], dtype="f")

    b0 = np.array([
        [1.],
        [1.],
        [1.],
        [1.]], dtype="f")

    W1 = np.array([
        [0., 2., 1., 0.],
        [1., 0., 2., 1.],
        [2., 1., 0., 2.],
        [0., 2., 1., 0.],
        [1., 0., 2., 1.],
        [2., 1., 0., 2.],
        [0., 2., 1., 0.],
        [1., 0., 2., 1.]], dtype="f")

    b1 = np.array([
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.]], dtype="f")

    W2 = np.array([
        [0., 2., 1., 0., 2., 1., 0., 2.],
        [1., 0., 2., 1., 0., 2., 1., 0.],
        [2., 1., 0., 2., 1., 0., 2., 1.],
        [0., 2., 1., 0., 2., 1., 0., 2.],
        [1., 0., 2., 1., 0., 2., 1., 0.],
        [2., 1., 0., 2., 1., 0., 2., 1.],
        [0., 2., 1., 0., 2., 1., 0., 2.],
        [1., 0., 2., 1., 0., 2., 1., 0.]], dtype="f")

    b2 = np.array([
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.]], dtype="f")

    W3 = np.array([
        [0., 1., 2., 0., 1., 2., 0., 1.],
        [1., 2., 0., 1., 2., 0., 1., 2.],
        [2., 0., 1., 2., 0., 1., 2., 0.],
        [0., 1., 2., 0., 1., 2., 0., 1.]], dtype="f")

    b3 = np.array([
        [1.],
        [1.],
        [1.],
        [1.]], dtype="f")

    W4 = np.array([
        [0., 2., 1., 0.],
        [1., 0., 2., 1.]], dtype="f")

    b4 = np.array([
        [1.],
        [1.]], dtype="f")

    mlp4.layers[0].W = W0
    mlp4.layers[0].b = b0
    mlp4.layers[2].W = W1
    mlp4.layers[2].b = b1
    mlp4.layers[4].W = W2
    mlp4.layers[4].b = b2
    mlp4.layers[6].W = W3
    mlp4.layers[6].b = b3
    mlp4.layers[8].W = W4
    mlp4.layers[8].b = b4

    A5 = mlp4.forward(A0)
    return mlp4

def MLP4_backward(mlp4):

    dLdA5 = np.array([
        [-4., -3.],
        [-2., -1.],
        [ 0.,  1.],
        [ 2.,  3.],
        [ 4.,  5.]], dtype="f")

    mlp4.backward(dLdA5)

def reset_prng():
    np.random.seed(11785)

base_dir = "./test_adam_dropout"

###############################################################################

print_header("1.1 Test Adam")

reset_prng()

try:
    mlp4 = MLP4(debug=True)
    optimizer = Adam(
        mlp4, 0.008, beta1=0.9, beta2=0.999, eps=1e-8,
    )
    num_test_updates = 5
    for _ in range(num_test_updates):
        MLP4_forward(mlp4)
        MLP4_backward(mlp4)
        optimizer.step()

    W = [x.W.round(4) for x in optimizer.l]
    b = [x.b.round(4) for x in optimizer.l]

    with open(f"{base_dir}/adam_sol_W.pkl", "rb") as stream:
        test_W = pickle.load(stream)

    with open(f"{base_dir}/adam_sol_b.pkl", "rb") as stream:
        test_b = pickle.load(stream)

    TEST_adam_W = True
    for idx in range(len(W)):
        TEST_adam_W &= np.allclose(W[idx], test_W[idx])

    print_header("[STUDENT OUTPUT]  Adam - weights after update")
    for idx in range(len(W)):
        print_header(f"Weight {idx+1}")
        print_array_details(W[idx])

    print_header("[EXPECTED OUTPUT]  Adam - weights after update")
    for idx in range(len(test_W)):
        print_header(f"Weight {idx+1}")
        print_array_details(test_W[idx])

    TEST_adam_b = True
    for idx in range(len(b)):
        TEST_adam_b &= np.allclose(b[idx], test_b[idx])

    print_header("[STUDENT OUTPUT]  Adam - bias after update")
    for idx in range(len(b)):
        print_header(f"Bias {idx+1}")
        print_array_details(b[idx])

    print_header("[EXPECTED OUTPUT]  Adam - bias after update")
    for idx in range(len(test_b)):
        print_header(f"Bias {idx+1}")
        print_array_details(test_b[idx])

except Exception as ex:
    print_header(f"Error in Test Adam: {str(ex.args)}")
    TEST_adam_W = False
    TEST_adam_b = False

print_status_message(
    test_number = "1.1  ",
    check_result = TEST_adam_W & TEST_adam_b, 
    description = "Adam"
)


###############################################################################

print_header("1.2 Test AdamW")

reset_prng()

try:
    mlp4 = MLP4(debug=True)
    optimizer = AdamW(
        mlp4, 0.008, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01,
    )
    num_test_updates = 5
    for _ in range(num_test_updates):
        MLP4_forward(mlp4)
        MLP4_backward(mlp4)
        optimizer.step()

    W = [x.W.round(4) for x in optimizer.l]
    b = [x.b.round(4) for x in optimizer.l]

    with open(f"{base_dir}/adamW_sol_W.pkl", "rb") as stream:
        test_W = pickle.load(stream)

    with open(f"{base_dir}/adamW_sol_b.pkl", "rb") as stream:
        test_b = pickle.load(stream)

    TEST_adamW_W = True
    for idx in range(len(W)):
        TEST_adamW_W &= np.allclose(W[idx], test_W[idx])

    print_header("[STUDENT OUTPUT]  AdamW - weights after update")
    for idx in range(len(W)):
        print_header(f"Weight {idx+1}")
        print_array_details(W[idx])

    print_header("[EXPECTED OUTPUT]  AdamW - weights after update")
    for idx in range(len(test_W)):
        print_header(f"Weight {idx+1}")
        print_array_details(test_W[idx])

    TEST_adamW_b = True
    for idx in range(len(b)):
        TEST_adamW_b &= np.allclose(b[idx], test_b[idx])

    print_header("[STUDENT OUTPUT]  AdamW - bias after update")
    for idx in range(len(b)):
        print_header(f"Bias {idx+1}")
        print_array_details(b[idx])

    print_header("[EXPECTED OUTPUT]  AdamW - bias after update")
    for idx in range(len(test_b)):
        print_header(f"Bias {idx+1}")
        print_array_details(test_b[idx])

except Exception as ex:
    print_header(f"Error in Test AdamW: {str(ex.args)}")
    TEST_adamW_W = False
    TEST_adamW_b = False

print_status_message(
    test_number = "1.2  ",
    check_result = TEST_adamW_W & TEST_adamW_b, 
    description = "AdamW"
)

###############################################################################

print_header("2.1 Test Dropout Forward")

try:
    x = np.random.randn(20, 64)

    reset_prng()

    dropout_layer = mytorch.nn.dropout.Dropout(p=0.3)
    y = dropout_layer(x)

    with open(f"{base_dir}/dropout_sol_forward.pkl", "rb") as stream:
        test_dropout_forward = pickle.load(stream)

    TEST_dropout_forward = np.allclose(y.round(4), test_dropout_forward)

    print_header("[STUDENT OUTPUT]  Dropout Forward")
    print_array_details(y.round(4))

    print_header("[EXPECTED OUTPUT]  Dropout Forward")
    print_array_details(test_dropout_forward)

except Exception as ex:
    print_header(f"Error in Dropout Forward: {str(ex.args)}")
    TEST_dropout_forward = False

print_status_message(
    test_number = "2.1    ",
    check_result = TEST_dropout_forward, 
    description = "Dropout Forward"
)

###############################################################################

print_header("2.2 Test Dropout Backward")

reset_prng()

try:
    x = np.random.randn(20, 64)

    reset_prng()

    dropout_layer = mytorch.nn.dropout.Dropout(p=0.3)
    y = dropout_layer(x)

    reset_prng()

    delta = np.random.randn(20, 64)
    dx = dropout_layer.backward(delta)

    with open(f"{base_dir}/dropout_sol_backward.pkl", "rb") as stream:
        test_dropout_backward = pickle.load(stream)

    TEST_dropout_backward = np.allclose(dx.round(4), test_dropout_backward)

    print_header("[STUDENT OUTPUT]  Dropout Backward")
    print_array_details(dx.round(4))

    print_header("[EXPECTED OUTPUT]  Dropout Backward")
    print_array_details(test_dropout_backward)

except Exception as ex:
    print_header(f"Error in Dropout Backward: {str(ex.args)}")
    TEST_dropout_backward = False

print_status_message(
    test_number = "2.2    ",
    check_result = TEST_dropout_backward, 
    description = "Dropout Backward"
)

###############################################################################

SCORE_Adam = 5. * int(TEST_adam_W & TEST_adam_b)
SCORE_AdamW = 5. * int(TEST_adamW_W & TEST_adamW_b)
SCORE_Dropout_Forward = 5. * int(TEST_dropout_forward)
SCORE_Dropout_Backward = 5. * int(TEST_dropout_backward)
SCORE_Total = 20.

###############################################################################

GRADE_LOGS = {
    "Adam": SCORE_Adam,
    "AdamW": SCORE_AdamW,
    "Dropout Forward": SCORE_Dropout_Forward,
    "Dropout Backward": SCORE_Dropout_Backward,
}

###############################################################################

print("\n")
print("TEST    | STATUS | SCORE | DESCRIPTION")
print("────────┼────────┼───────┼──────────────────────────────────")

for i, (key, value) in enumerate(GRADE_LOGS.items()):
    
    index_str = str(i).zfill(2)
    
    if value == 0:
        status_str = " │\033[31m FAILED \033[0m│ "
    else:
        status_str = " │\033[32m PASSED \033[0m│ "

    score = GRADE_LOGS.get(key)
    if score > 0.0:
        score_str = f"\033[32m {score}  \033[0m│ "
    else:
        score_str = f"\033[31m {score}  \033[0m│ "
    
    print("Test ", index_str, status_str, score_str, key, sep="")

print("\n")

print(f"Total Score: {sum(list(GRADE_LOGS.values()))} / {SCORE_Total}\n")

print("\n")

print(json.dumps({'scores': GRADE_LOGS}))

