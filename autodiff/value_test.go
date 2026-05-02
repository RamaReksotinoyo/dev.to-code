package main

import (
	"fmt"
	"math"
	"testing"
)

const tol = 1e-6
const tolFD = 1e-4 // finite difference tolerance

// Computes finiteDiff numerical derivative: (f(x+h) - f(x-h)) / 2h
// Used to verify that our autodiff implementation is correct.
func finiteDiff(f func(float64) float64, x float64) float64 {
	h := 1e-5
	return (f(x+h) - f(x-h)) / (2 * h)
}

func assertClose(t *testing.T, got, want, tolerance float64, name string) {
	t.Helper()
	if math.Abs(got-want) > tolerance {
		t.Errorf("%s: got=%.8f want=%.8f diff=%.2e", name, got, want, math.Abs(got-want))
	}
}


func TestAdd(t *testing.T) {
	// z = x + y
	// dz/dx = 1, dz/dy = 1
	x, y := New(2), New(3)
	z := Add(x, y)
	z.Backward()

	assertClose(t, z.Data, 5, tol, "add forward")
	assertClose(t, x.Grad, 1, tol, "add dx=1")
	assertClose(t, y.Grad, 1, tol, "add dy=1")
}

func TestMul(t *testing.T) {
	// z = x * y
	// dz/dx = y = 3, dz/dy = x = 2
	x, y := New(2), New(3)
	z := Mul(x, y)
	z.Backward()

	assertClose(t, z.Data, 6, tol, "mul forward")
	assertClose(t, x.Grad, 3, tol, "mul dx=y=3")
	assertClose(t, y.Grad, 2, tol, "mul dy=x=2")
}

func TestSub(t *testing.T) {
	x, y := New(5), New(3)
	z := Sub(x, y)
	z.Backward()

	assertClose(t, z.Data, 2, tol, "sub forward")
	assertClose(t, x.Grad, 1, tol, "sub dx=1")
	assertClose(t, y.Grad, -1, tol, "sub dy=-1")
}

func TestDiv(t *testing.T) {
	x, y := New(6), New(2)
	z := Div(x, y)
	z.Backward()

	assertClose(t, z.Data, 3, tol, "div forward")
	assertClose(t, x.Grad, 0.5, tol, "div dx=1/y=0.5")
	assertClose(t, y.Grad, -1.5, tol, "div dy=-x/y²=-1.5")
}

func TestPow(t *testing.T) {
	// z = x^3, dz/dx = 3x² = 12 when x=2
	x := New(2)
	z := Pow(x, 3)
	z.Backward()

	assertClose(t, z.Data, 8, tol, "pow forward 2³=8")
	assertClose(t, x.Grad, 12, tol, "pow dx=3x²=12")
}

func TestTanh(t *testing.T) {
	// z = tanh(x), dz/dx = 1 - tanh(x)²
	xv := 0.5
	x := New(xv)
	z := Tanh(x)
	z.Backward()

	t_ := math.Tanh(xv)
	assertClose(t, z.Data, t_, tol, "tanh forward")
	assertClose(t, x.Grad, 1-t_*t_, tol, "tanh dx=1-tanh²")
}

func TestReLU(t *testing.T) {
	// x > 0: gradient flows
	x := New(2.0)
	z := ReLU(x)
	z.Backward()
	assertClose(t, z.Data, 2.0, tol, "relu x>0 forward")
	assertClose(t, x.Grad, 1.0, tol, "relu x>0 dx=1")

	// x < 0: neuron is "dead", gradient does not flow
	x2 := New(-1.5)
	z2 := ReLU(x2)
	z2.Backward()
	assertClose(t, z2.Data, 0.0, tol, "relu x<0 forward=0")
	assertClose(t, x2.Grad, 0.0, tol, "relu x<0 dx=0 (mati)")
}

func TestExp(t *testing.T) {
	x := New(1.0)
	z := Exp(x)
	z.Backward()

	assertClose(t, z.Data, math.E, tol, "exp forward")
	assertClose(t, x.Grad, math.E, tol, "exp dx=e^x")
}

func TestLog(t *testing.T) {
	x := New(2.0)
	z := Log(x)
	z.Backward()

	assertClose(t, z.Data, math.Log(2), tol, "log forward")
	assertClose(t, x.Grad, 0.5, tol, "log dx=1/x=0.5")
}

// Shared variable test. This verifies the += accumulation in backward.

func TestSharedVariable(t *testing.T) {
	// L = x² + x  (x is used twice)
	// dL/dx = 2x + 1 = 7 when x=3
	//
	// The gradient for x comes from TWO paths:
	//   - from x*x: contribution is 2x = 6
	//   - from +x:  contribution is 1
	// Total: 7  <- this is why the code uses += instead of =
	x := New(3)
	L := Add(Mul(x, x), x)
	L.Backward()

	assertClose(t, L.Data, 12, tol, "x²+x forward: 9+3=12")
	assertClose(t, x.Grad, 7, tol, "x²+x dx=2x+1=7")
}

func TestSharedVariableCube(t *testing.T) {
	// L = x * x * x  (= x³)
	// dL/dx = 3x² = 12 when x=2
	x := New(2)
	L := Mul(Mul(x, x), x)
	L.Backward()

	assertClose(t, L.Data, 8, tol, "x³ forward: 8")
	assertClose(t, x.Grad, 12, tol, "x³ dx=3x²=12")
}

// Test aturan rantai panjang

func TestChainRule(t *testing.T) {
	// L = tanh(x*w + b)
	//
	// Forward:
	//   s = x*w = 1*0.5 = 0.5
	//   h = s+b = 0.5+0.1 = 0.6
	//   L = tanh(0.6)
	//
	// Backward (chain rule loop):
	//   dL/dh = 1 - tanh(0.6)²
	//   dL/ds = dL/dh * 1          (from add: dh/ds=1)
	//   dL/dw = dL/ds * x = dL/dh * 1.0
	//   dL/dx = dL/ds * w = dL/dh * 0.5
	//   dL/db = dL/dh * 1
	xv, wv, bv := 1.0, 0.5, 0.1
	x, w, b := New(xv), New(wv), New(bv)
	L := Tanh(Add(Mul(x, w), b))
	L.Backward()

	hv := xv*wv + bv
	dtanh := 1 - math.Tanh(hv)*math.Tanh(hv)

	assertClose(t, L.Data, math.Tanh(hv), tol, "chain forward")
	assertClose(t, w.Grad, dtanh*xv, tol, "chain dL/dw")
	assertClose(t, x.Grad, dtanh*wv, tol, "chain dL/dx")
	assertClose(t, b.Grad, dtanh, tol, "chain dL/db")
}

// Gradient check. autodiff vs finite difference

func TestGradientCheck(t *testing.T) {
	// f(x) = x³ - 2x + 1
	// f'(x) = 3x² - 2
	buildGraph := func(xv float64) (*Value, *Value) {
		x := New(xv)
		x3 := Pow(x, 3)
		cx := Mul(New(2), x)
		L := Add(Sub(x3, cx), New(1))
		return x, L
	}

	fPoly := func(xv float64) float64 {
		return xv*xv*xv - 2*xv + 1
	}

	testPoints := []float64{-2.0, -0.5, 0.0, 1.0, 3.0}
	for _, xv := range testPoints {
		x, L := buildGraph(xv)
		L.Backward()

		analytical := 3*xv*xv - 2
		numerical := finiteDiff(fPoly, xv)

		assertClose(t, x.Grad, analytical, 1e-9,
			"grad_check analitik x="+formatFloat(xv))
		assertClose(t, x.Grad, numerical, tolFD,
			"grad_check finite_diff x="+formatFloat(xv))
	}
}

func formatFloat(f float64) string {
	if f == math.Trunc(f) {
		return fmt.Sprintf("%.0f", f)
	}
	return fmt.Sprintf("%.1f", f)
}

// Test MSE loss. One-step gradient descent simulation

func TestMSELoss(t *testing.T) {
	// Model: y_pred = tanh(x*w + b)
	// Loss:  L = (y_pred - y_true)²
	//
	// Ini yang terjadi di setiap iterasi training loop:
	// hitung loss -> backward -> update w dan b
	x := New(1.5)
	w := New(0.3)
	b := New(-0.1)
	yTrue := New(1.0)

	yPred := Tanh(Add(Mul(x, w), b))
	diff := Sub(yPred, yTrue)
	loss := Mul(diff, diff)
	loss.Backward()

	// Verifikasi dL/dw dengan finite difference
	fMSE := func(wv float64) float64 {
		p := math.Tanh(1.5*wv + (-0.1))
		return (p - 1.0) * (p - 1.0)
	}
	fdW := finiteDiff(fMSE, 0.3)
	assertClose(t, w.Grad, fdW, tolFD, "MSE dL/dw vs finite diff")
}

func TestZeroGrad(t *testing.T) {
	x := New(2.0)
	L := Mul(x, x)
	L.Backward()
	assertClose(t, x.Grad, 4.0, tol, "sebelum zero: grad=4")

	L.ZeroGrad()
	assertClose(t, x.Grad, 0.0, tol, "setelah zero: grad=0")
}


// BenchmarkForwardMul measures the cost of constructing a single Mul node.
func BenchmarkForwardMul(b *testing.B) {
	x, y := New(2.0), New(3.0)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Mul(x, y)
	}
}
 
// BenchmarkForwardChain measures the forward pass for L = tanh(x*w + b).
// This is the most common computational unit in a single neuron.
func BenchmarkForwardChain(b *testing.B) {
	x, w, bc := New(1.0), New(0.5), New(0.1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Tanh(Add(Mul(x, w), bc))
	}
}
 
// BenchmarkBackwardChain measures a full cycle: forward + backward.
// This is what happens in each iteration of the training loop.
func BenchmarkBackwardChain(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x, w, bc := New(1.0), New(0.5), New(0.1)
		L := Tanh(Add(Mul(x, w), bc))
		L.Backward()
	}
}
 
// BenchmarkBackwardMSE measures full cycle MSE loss:
//   y_pred = tanh(x*w + b)
//   loss   = (y_pred - y_true)²
func BenchmarkBackwardMSE(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x, w, bc := New(1.5), New(0.3), New(-0.1)
		yTrue := New(1.0)
		yPred := Tanh(Add(Mul(x, w), bc))
		diff := Sub(yPred, yTrue)
		loss := Mul(diff, diff)
		loss.Backward()
	}
}
 
// BenchmarkBackwardDeepChain measures long computional chain:
//   L = tanh(tanh(tanh(x*w + b)))
// Lebih banyak node = topological sort lebih dalam.
func BenchmarkBackwardDeepChain(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x, w, bc := New(1.0), New(0.5), New(0.1)
		h1 := Tanh(Add(Mul(x, w), bc))
		h2 := Tanh(Add(Mul(h1, w), bc))
		h3 := Tanh(Add(Mul(h2, w), bc))
		h3.Backward()
	}
}
 
// BenchmarkZeroGrad measures gradien reset consts.
func BenchmarkZeroGrad(b *testing.B) {
	x, w, bc := New(1.0), New(0.5), New(0.1)
	L := Tanh(Add(Mul(x, w), bc))
	L.Backward()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		L.ZeroGrad()
	}
}
