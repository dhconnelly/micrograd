use std::{borrow::BorrowMut, rc::Rc};

pub struct Value {
    val: f64,
    label: String,
    prev: Vec<ValueRef>,
}

impl Value {
    pub fn of(val: f64, label: impl Into<String>) -> ValueRef {
        let label = label.into();
        ValueRef(Rc::new(Value { val, label, prev: Vec::new() }))
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Value(label='{}', val='{}')", self.label, self.val)
    }
}

pub struct ValueRef(Rc<Value>);

impl std::fmt::Display for ValueRef {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl std::ops::Deref for ValueRef {
    type Target = Value;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl ValueRef {
    pub fn tanh(self) -> ValueRef {
        let e2x = f64::exp(2.0 * self.val);
        let val = (e2x - 1.0) / (e2x + 1.0);
        let label = format!("tanh({})", self.label);
        let prev = vec![self];
        ValueRef(Rc::new(Value { val, label, prev }))
    }
}

impl std::ops::Add for ValueRef {
    type Output = ValueRef;
    fn add(self, rhs: ValueRef) -> ValueRef {
        let val = self.val + rhs.val;
        let label = format!("{} + {}", self.label, rhs.label);
        let prev = vec![self, rhs];
        ValueRef(Rc::new(Value { val, label, prev }))
    }
}

impl std::ops::Mul for ValueRef {
    type Output = ValueRef;
    fn mul(self, rhs: ValueRef) -> ValueRef {
        let val = self.val * rhs.val;
        let label = format!("{}*{}", self.label, rhs.label);
        let prev = vec![self, rhs];
        ValueRef(Rc::new(Value { val, label, prev }))
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
