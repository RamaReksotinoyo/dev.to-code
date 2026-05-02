package main

import (
	"fmt"
)

func main() {

	x_test := New(1.0)
	w_test := New(0.5)
	b_test := New(0.1)

	s_test := Mul(x_test, w_test)
	L := Add(s_test, b_test)
	L.Backward()

	fmt.Println(w_test.Grad) // 1.0
	fmt.Println(x_test.Grad) // 0.5
	fmt.Println(b_test.Grad) // 1.0

	// xs := []float64{1.0, -1.0, 1.0, -1.0}
	// ys := []float64{1.0, -1.0, 1.0, -1.0}
	//
	// w := New(0.1)
	// b := New(0.0)
	// lr := 0.1
	//
	// fmt.Println("Melatih neuron: y_pred = tanh(x*w + b)")
	// fmt.Printf("%-6s  %-12s  %-10s  %-10s\n", "Epoch", "Loss", "w", "b")
	// fmt.Println("----------------------------------------------")
	//
	// for epoch := 0; epoch < 50; epoch++ {
	// 	totalLoss := 0.0
	// 	gradW := 0.0
	// 	gradB := 0.0
	//
	// 	for i, xv := range xs {
	// 		x := New(xv)
	// 		yTrue := New(ys[i])
	//
	// 		yPred := Tanh(Add(Mul(x, w), b))
	// 		diff := Sub(yPred, yTrue)
	// 		loss := Mul(diff, diff)
	//
	// 		loss.Backward()
	//
	// 		totalLoss += loss.Data
	// 		gradW += w.Grad
	// 		gradB += b.Grad
	//
	// 		loss.ZeroGrad()
	// 	}
	//
	// 	w.Data -= lr * gradW / float64(len(xs))
	// 	b.Data -= lr * gradB / float64(len(xs))
	//
	// 	if epoch%5 == 0 || epoch == 49 {
	// 		fmt.Printf("%-6d  %-12.6f  %-10.6f  %-10.6f\n",
	// 			epoch, totalLoss/float64(len(xs)), w.Data, b.Data)
	// 	}
	// }
	//
	// fmt.Println("\nHasil akhir:")
	// for i, xv := range xs {
	// 	pred := math.Tanh(xv*w.Data + b.Data)
	// 	fmt.Printf("  x=%+.1f  →  y_pred=%+.4f  (target=%+.1f)\n", xv, pred, ys[i])
	// }
}
