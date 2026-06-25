# TinyML v0.1.0-beta
- Machine learning in v language.
- Just a hot-take on ml logic.

## Installazation:
```sh
v install h-sys-2025.tinyML
```

## Example: Xor and And.
```v
module main

import h_sys_2025.tinyml.tinyml as ml


fn xor_model() {
    mut my_nn := ml.new_nn(2, 4, 1)  // 2 inputs, 4 hidden neurons, 1 output

    inputs := [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]

    targets := [
        [0.0],
        [1.0],
        [1.0],
        [1.0],
    ]

    println('Training XOR-like problem...')
    my_nn.train_epochs(inputs, targets, 20000)

    // Test predictions
    println('Results:')
    for i, inp in inputs {
        pred := my_nn.predict(inp)
        println('${inp} -> ${pred[0]:.4f}  (target: ${targets[i][0]})')
    }
}


fn and_model() {
    mut my_nn := ml.new_nn(2, 10, 1)  // 2 inputs, 4 hidden neurons, 1 output

    inputs := [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]

    targets := [
        [0.0],
        [1.0],
        [1.0],
        [1.0],
    ]

    println('Training AND problem...')
    my_nn.train_epochs(inputs, targets, 20000)

    // Test predictions
    println('Results:')
    for i, inp in inputs {
        pred := my_nn.predict(inp)
        println('${inp} -> ${pred[0]:.4f}  (target: ${targets[i][0]})')
    }
}

fn main() {
    and_model()
}
```