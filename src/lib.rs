// TODO: shouldn't take ownership of |prev|. clone? refs?
#[derive(Clone)]
pub struct Value {
    val: f64,
    label: String,
    prev: Vec<Value>,
}

impl Value {
    pub fn of(val: f64, label: impl Into<String>) -> Value {
        Value { val, label: label.into(), prev: Vec::new() }
    }

    pub fn tanh(self) -> Value {
        let e2x = f64::exp(2.0 * self.val);
        let val = (e2x - 1.0) / (e2x + 1.0);
        let label = format!("tanh({})", self.label);
        let mut result = Value::of(val, label);
        result.prev = vec![self];
        result
    }
}

impl std::ops::Mul for Value {
    type Output = Value;
    fn mul(self, rhs: Self) -> Self::Output {
        let label = format!("{}*{}", self.label, rhs.label);
        let mut prod = Value::of(self.val * rhs.val, label);
        prod.prev = vec![self, rhs];
        prod
    }
}

impl std::ops::Add for Value {
    type Output = Value;
    fn add(self, rhs: Self) -> Self::Output {
        let label = format!("{} + {}", self.label, rhs.label);
        let mut sum = Value::of(self.val + rhs.val, label);
        sum.prev = vec![self, rhs];
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn float_equal(x: f64, y: f64) -> bool {
        (x - y).abs() < f64::EPSILON
    }

    #[test]
    fn test_forward() {
        let x1 = Value::of(2.0, "x1");
        let x2 = Value::of(0.0, "x2");
        let w1 = Value::of(-3.0, "w1");
        let w2 = Value::of(1.0, "w2");
        let b = Value::of(6.881373587019543, "b");

        let x1w1 = x1 * w1;
        assert!(float_equal(x1w1.val, -6.0));

        let x2w2 = x2 * w2;
        assert!(float_equal(x2w2.val, 0.0));

        let x1w1x2w2 = x1w1 + x2w2;
        assert!(float_equal(x1w1x2w2.val, -6.0));

        let n = x1w1x2w2 + b;
        assert!(float_equal(n.val, 0.881373587019543));
        assert_eq!(n.label, String::from("x1*w1 + x2*w2 + b"));

        let o = n.tanh();
        assert_eq!(o.label, String::from("tanh(x1*w1 + x2*w2 + b)"));
    }
}
