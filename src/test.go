package main

import (
	"ANN"
	"fmt"
)

func main() {

	network := ANN.MakeSimple(3, 3, 2, "BPSig")
	network.Print()
	network.RandomWeights(-.05, .05)
	network.Print()

	answer, err := network.Evaluate([]float64{1, 2, 3})

	if err != nil {
		fmt.Println(err)
		panic("panicc!!!")
	}

	err = network.BackProp([]float64{1, 2, 3}, []float64{1, 2, 3}, .2)

	if err != nil {
		fmt.Println(err)
		panic("panicc!!!")
	}

	fmt.Println(answer, err)

	network.Print()

}
