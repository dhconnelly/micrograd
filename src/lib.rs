use std::{
    cell::{Ref, RefCell},
    collections::HashSet,
    rc::Rc,
};

#[derive(Debug)]
pub enum Op {
    None,
    Add(ValueRef, ValueRef),
    Mul(ValueRef, ValueRef),
    Tanh(ValueRef),
}

#[derive(Debug)]
pub struct Value {
    val: f64,
    grad: f64,
    label: String,
    op: Op,
}

impl Value {
    pub fn of(val: f64, label: impl Into<String>) -> ValueRef {
        let label = label.into();
        let grad = 0.0;
        let op = Op::None;
        Value { val, grad, label, op }.into()
    }

    fn with_op(val: f64, label: String, op: Op) -> ValueRef {
        Value { val, grad: 0.0, label, op }.into()
    }
}

impl Into<ValueRef> for Value {
    fn into(self) -> ValueRef {
        ValueRef(Rc::new(RefCell::new(self)))
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "[ {} | val = {} | grad = {} ]",
            self.label, self.val, self.grad
        )
    }
}

#[derive(Clone, Debug)]
pub struct ValueRef(Rc<RefCell<Value>>);

impl std::fmt::Display for ValueRef {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.0.borrow().fmt(f)
    }
}

fn topological_sort(root: ValueRef) -> Vec<ValueRef> {
    fn rec(val: ValueRef, v: &mut HashSet<String>, sorted: &mut Vec<ValueRef>) {
        // can we avoid using the labels for the set here?
        if !v.contains(val.label().as_str()) {
            v.insert(val.label().to_string());
            for prev in val.prev() {
                // TODO: eliminate clone
                rec(prev.clone(), v, sorted);
            }
            sorted.push(val);
        }
    }
    let mut sorted = Vec::new();
    rec(root, &mut HashSet::new(), &mut sorted);
    sorted
}

impl ValueRef {
    pub fn tanh(&self) -> ValueRef {
        let e2x = f64::exp(2.0 * self.val());
        let val = (e2x - 1.0) / (e2x + 1.0);
        let label = format!("tanh({})", self.label());
        Value::with_op(val, label, Op::Tanh(self.clone()))
    }

    // using operator overloading would require taking references everywhere,
    // e.g. &x + &y because we can't make ValueRef Copy :(
    pub fn mul(&self, rhs: &ValueRef) -> ValueRef {
        let val = self.val() * rhs.val();
        let label = format!("{}*{}", self.label(), rhs.label());
        Value::with_op(val, label, Op::Mul(self.clone(), rhs.clone()))
    }

    pub fn add(&self, rhs: &ValueRef) -> ValueRef {
        let val = self.val() + rhs.val();
        let label = format!("{} + {}", self.label(), rhs.label());
        Value::with_op(val, label, Op::Add(self.clone(), rhs.clone()))
    }

    pub fn val(&self) -> f64 {
        self.0.borrow().val
    }

    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }

    pub fn label<'a>(&'a self) -> Ref<'a, String> {
        Ref::map(self.0.borrow(), |r| &r.label)
    }

    fn local_backward(&mut self) {
        match &self.0.borrow().op {
            Op::None => {}
            Op::Add(lhs, rhs) => {
                lhs.0.borrow_mut().grad += self.grad();
                rhs.0.borrow_mut().grad += self.grad();
            }
            Op::Mul(lhs, rhs) => {
                lhs.0.borrow_mut().grad += rhs.val() * self.grad();
                rhs.0.borrow_mut().grad += lhs.val() * self.grad();
            }
            Op::Tanh(arg) => {
                let e2x = f64::exp(2.0 * arg.val());
                let t = (e2x - 1.0) / (e2x + 1.0);
                arg.0.borrow_mut().grad += (1.0 - t.powf(2.0)) * self.grad();
            }
        }
    }

    pub fn backward(&mut self) {
        self.0.borrow_mut().grad = 1.0;
        let mut nodes = topological_sort(self.clone());
        nodes.reverse();
        for mut node in nodes {
            node.local_backward();
        }
    }

    fn prev(&self) -> Vec<ValueRef> {
        match &self.0.borrow().op {
            Op::None => vec![],
            Op::Add(lhs, rhs) | Op::Mul(lhs, rhs) => {
                vec![lhs.clone(), rhs.clone()]
            }
            Op::Tanh(arg) => vec![arg.clone()],
        }
    }
}

pub fn trace(val: &ValueRef) {
    fn trace(val: &ValueRef, v: &mut HashSet<String>, n: usize) {
        if !v.contains(val.label().as_str()) {
            v.insert(val.label().to_string());
            let padding = "|   ".repeat(n);
            println!("{}{}", padding, val);
            for child in &val.prev() {
                trace(child, v, n + 1);
            }
        }
    }
    trace(val, &mut HashSet::new(), 0);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn float_equal(x: f64, y: f64) -> bool {
        (x - y).abs() < 0.001
    }

    #[test]
    fn test_forward() {
        let x1 = Value::of(2.0, "x1");
        let x2 = Value::of(0.0, "x2");
        let w1 = Value::of(-3.0, "w1");
        let w2 = Value::of(1.0, "w2");
        let b = Value::of(6.881373587019543, "b");
        let x1w1 = x1.mul(&w1);
        let x2w2 = x2.mul(&w2);
        let x1w1x2w2 = x1w1.add(&x2w2);
        let n = x1w1x2w2.add(&b);
        let o = n.tanh();

        assert!(float_equal(x1w1.val(), -6.0));
        assert!(float_equal(x2w2.val(), 0.0));
        assert!(float_equal(x1w1x2w2.val(), -6.0));
        assert!(float_equal(n.val(), 0.881373587019543));
        assert!(float_equal(o.val(), 0.7071));
    }

    #[test]
    fn test_backward() {
        let x1 = Value::of(2.0, "x1");
        let x2 = Value::of(0.0, "x2");
        let w1 = Value::of(-3.0, "w1");
        let w2 = Value::of(1.0, "w2");
        let b = Value::of(6.881373587019543, "b");
        let x1w1 = x1.mul(&w1);
        let x2w2 = x2.mul(&w2);
        let x1w1x2w2 = x1w1.add(&x2w2);
        let n = x1w1x2w2.add(&b);
        let mut o = n.tanh();
        o.backward();

        assert!(float_equal(x1.grad(), -1.5));
        assert!(float_equal(w1.grad(), 1.0));
        assert!(float_equal(x2.grad(), 0.5));
        assert!(float_equal(w2.grad(), 0.0));
    }
}
