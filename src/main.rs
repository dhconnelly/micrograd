use micrograd::Value;

fn main() {
    let x1 = Value::of(2.0, "x1");
    let x2 = Value::of(0.0, "x2");
    let w1 = Value::of(-3.0, "w1");
    let w2 = Value::of(1.0, "w2");
    let b = Value::of(6.881373587019543, "b");
    let x1w1 = x1 * w1;
    let x2w2 = x2 * w2;
    let x1w1x2w2 = x1w1 + x2w2;
    let n = x1w1x2w2 + b;
    let o = n.tanh();
    println!("{}", o);
}
