# micrograd

a rust port of andrej karpathy's https://github.com/karpathy/micrograd

# why

an exercise while following along with https://www.youtube.com/watch?v=VMj-3S1tku0

# who

daniel connelly <dhconnelly@gmail.com> (https://dhconnelly.com)

# example

```rust
use micrograd::nn::{flatten, from_tensor, rmse, tensor, Tensor, MLP};

fn train_mlp(xs: &[Tensor], ys: &Tensor, step: f64, iterations: usize) -> MLP {
    let mut nn = MLP::new(3, 4, vec![4, 1]);
    for i in 0..iterations {
        // forward pass
        let ypred = flatten(nn.forward2(&xs));
        let mut loss = rmse(&ypred, &ys);

        // backward pass
        for mut p in nn.parameters() {
            p.zero_grad();
        }
        loss.backward();

        // update parameters
        for mut p in nn.parameters() {
            p.adjust_val(step * -p.grad());
        }

        if i % 10 == 0 {
            println!("{:5} loss {}", i, loss.val());
        }
    }
    nn
}

fn main() {
    let xs = [
        tensor([2.0, 3.0, -1.0]),
        tensor([3.0, -1.0, 0.5]),
        tensor([0.5, 1.0, 1.0]),
        tensor([1.0, 1.0, -1.0]),
    ];
    let ys = tensor([1.0, -1.0, -1.0, 1.0]);
    let nn = train_mlp(&xs, &ys, 0.01, 200);
    let ypred = flatten(nn.forward2(&xs));
    println!("predictions: {:?}", from_tensor(&ypred));
    println!("rmse: {}", rmse(&ypred, &ys).val());
}
```

# license

MIT
