package main

import (
	"ANN"
	"fmt"
)

func main() {

	network := ANN.MakeSimple(3, 3, 2, "BPSig")
	network.RandomWeights(-.05, .05)

	answer, _ := network.Evaluate([]float64{1, 2, 3})
	fmt.Println(answer)
	answer, _ = network.Evaluate([]float64{3, 2, 1})
	fmt.Println(answer)

	for i := 0; i < 1000000; i++ {
		network.BackProp([]float64{1, 2, 3}, []float64{1, .2}, .3)
		network.BackProp([]float64{3, 2, 1}, []float64{-1, 1}, .3)

	}

	answer, _ = network.Evaluate([]float64{1, 2, 3})
	fmt.Println(answer)
	answer, _ = network.Evaluate([]float64{3, 2, 1})
	fmt.Println(answer)
}
