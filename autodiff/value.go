package main

import "math"

// Value a node in computational graph.
//
// Each value store:
//   - Data     : the result of the forward pass (actual value)
//   - Grad     : dL/d(self), populated during the backward pass
//   - backward : a closure that knows how to propagate gradients to its children
//   - children : input nodes that produced this Value
type Value struct {
	Data     float64
	Grad     float64
	backward func()
	children []*Value
}

// Create new leaf node, the input variable whose gradient we want to calculate.
func New(data float64) *Value {
	return &Value{Data: data}
}

// ============================================================
// Operation — Forward + Backward
//
// Pattern for each operation:
//   1. Compute the data (forward)
//   2. Save references to the input nodes
//   3. Define a backward() closure that applies the chain rule


// ============================================================

// Add: z = x + y
//
// Derivative:
//   ∂z/∂x = 1  →  dL/dx += dL/dz * 1
//   ∂z/∂y = 1  →  dL/dy += dL/dz * 1
func Add(x, y *Value) *Value {
	out := &Value{
		Data:     x.Data + y.Data,
		children: []*Value{x, y},
	}
	out.backward = func() {
		x.Grad += out.Grad
		y.Grad += out.Grad
	}
	return out
}

// Mul: z = x * y
//
// Derivative:
//   ∂z/∂x = y  →  dL/dx += dL/dz * y
//   ∂z/∂y = x  →  dL/dy += dL/dz * x
func Mul(x, y *Value) *Value {
	out := &Value{
		Data:     x.Data * y.Data,
		children: []*Value{x, y},
	}
	out.backward = func() {
		x.Grad += out.Grad * y.Data
		y.Grad += out.Grad * x.Data
	}
	return out
}

// Sub: z = x - y
//
// Derivative:
//   ∂z/∂x =  1  →  dL/dx += dL/dz
//   ∂z/∂y = -1  →  dL/dy -= dL/dz
func Sub(x, y *Value) *Value {
	out := &Value{
		Data:     x.Data - y.Data,
		children: []*Value{x, y},
	}
	out.backward = func() {
		x.Grad += out.Grad
		y.Grad -= out.Grad
	}
	return out
}

// Div: z = x / y
//
// Derivative:
//   ∂z/∂x =  1/y      →  dL/dx += dL/dz / y
//   ∂z/∂y = -x/y²     →  dL/dy -= dL/dz * x / y²
func Div(x, y *Value) *Value {
	out := &Value{
		Data:     x.Data / y.Data,
		children: []*Value{x, y},
	}
	out.backward = func() {
		x.Grad += out.Grad / y.Data
		y.Grad -= out.Grad * x.Data / (y.Data * y.Data)
	}
	return out
}

// Pow: z = x^exp  (The exponent is a constant, not a value)
//
// Derivative:
//   ∂z/∂x = exp * x^(exp-1)
//
// exp value is captured by closure
func Pow(x *Value, exp float64) *Value {
	out := &Value{
		Data:     math.Pow(x.Data, exp),
		children: []*Value{x},
	}
	out.backward = func() {
		x.Grad += out.Grad * exp * math.Pow(x.Data, exp-1)
	}
	return out
}

// Neg: z = -x
//
// Derivative:
//   ∂z/∂x = -1
func Neg(x *Value) *Value {
	out := &Value{
		Data:     -x.Data,
		children: []*Value{x},
	}
	out.backward = func() {
		x.Grad -= out.Grad
	}
	return out
}

// ReLU: z = max(0, x)
//
// Derivative:
//   ∂z/∂x = 1 jika x > 0, 0 jika x ≤ 0
//
// When x = 0, gradien = 0
func ReLU(x *Value) *Value {
	data := x.Data
	if data < 0 {
		data = 0
	}
	out := &Value{
		Data:     data,
		children: []*Value{x},
	}
	out.backward = func() {
		if x.Data > 0 {
			x.Grad += out.Grad
		}
	}
	return out
}

// Tanh: z = tanh(x)
//
// Derivative:
//   ∂z/∂x = 1 - tanh(x)²  =  1 - z²
//
// use out.Data (i.e., the already computed z) instead of
// recomputing tanh during backward. This is more efficient.
func Tanh(x *Value) *Value {
	t := math.Tanh(x.Data)
	out := &Value{
		Data:     t,
		children: []*Value{x},
	}
	out.backward = func() {
		x.Grad += out.Grad * (1 - out.Data*out.Data)
	}
	return out
}

// Exp: z = e^x
//
// Derivative:
//   ∂z/∂x = e^x = z
func Exp(x *Value) *Value {
	e := math.Exp(x.Data)
	out := &Value{
		Data:     e,
		children: []*Value{x},
	}
	out.backward = func() {
		x.Grad += out.Grad * out.Data
	}
	return out
}

// Log: z = ln(x)
//
// Derivative:
//   ∂z/∂x = 1/x
func Log(x *Value) *Value {
	out := &Value{
		Data:     math.Log(x.Data),
		children: []*Value{x},
	}
	out.backward = func() {
		x.Grad += out.Grad / x.Data
	}
	return out
}


// Backward computes gradients for all nodes connected to the root.
//
// Algorithm:
//  1. Topological sort. Ensure each node is processed only after all its descendants (so that gradients are fully accumulated when backward is called).
//  2. Set root.Grad = 1.0  (since dL/dL = 1)
//  3. Call backward() on each node, starting from the root down to the leaves.
func (root *Value) Backward() {
	// Collect all nodes in topological order
	topo := make([]*Value, 0)
	visited := make(map[*Value]bool)

	var buildTopo func(v *Value)
	buildTopo = func(v *Value) {
		if visited[v] {
			return
		}
		visited[v] = true
		for _, child := range v.children {
			buildTopo(child)
		}
		// Post-order: add node after all its children
		topo = append(topo, v)
	}
	buildTopo(root)

	// Traverse in reverse topological order (from root toward leaves)
	root.Grad = 1.0
	for i := len(topo) - 1; i >= 0; i-- {
		if topo[i].backward != nil {
			topo[i].backward()
		}
	}
}

// ZeroGrad resets all gradients in the subgraph to 0.
// Called before the next backward pass.
func (root *Value) ZeroGrad() {
	visited := make(map[*Value]bool)
	var zero func(v *Value)
	zero = func(v *Value) {
		if visited[v] {
			return
		}
		visited[v] = true
		v.Grad = 0
		for _, child := range v.children {
			zero(child)
		}
	}
	zero(root)
}
