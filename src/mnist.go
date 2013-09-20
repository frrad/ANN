package main

import (
	"ANN"
	"fmt"
	"io/ioutil"
	"strconv"
	"strings"
	"time"
)

const dim = 28

func readData(path string) map[[dim * dim]int]int {
	ans, err := ioutil.ReadFile(path)
	if err != nil {
		panic("ERROR READING FILE")
	}

	ret := make(map[[dim * dim]int]int)

	splat := strings.Split(string(ans), "\n")

	for _, data := range splat {
		split := strings.Split(data, ",")

		cheat, _ := strconv.Atoi(split[0])

		pixels := [dim * dim]int{}

		for i := 1; i < len(split); i++ {
			convert, _ := strconv.Atoi(split[i])
			pixels[i-1] = convert
		}

		ret[pixels] = cheat
	}

	return ret

}

func makeVec(number [dim * dim]int) []float64 {
	out := make([]float64, dim*dim)
	for i, pixel := range number {
		out[i] = (2. * float64(pixel) / 255.) - 1
	}
	return out
}

func makeOut(an int) []float64 {
	out := make([]float64, 10)
	for i := 0; i <= 9; i++ {
		if i == an {
			out[i] = 1
		} else {
			out[i] = -1
		}
	}
	return out
}

func pretty(vector []float64) {
	for x, val := range vector {
		val := (1 + val) * 5
		fmt.Print(x, ":")
		for i := 0; i < int(val); i++ {
			fmt.Print("+")
		}
		fmt.Print("\n")
	}
}

func main() {

	starttime := time.Now()

	training := readData("../data/train.csv")
	fmt.Println("Read in training data")

	network := ANN.MakeSimple(dim*dim, 300, 10, "BPSig")
	network.RandomWeights(-.01, .01)

	for j := 0; j < 60; j++ {

		//train

		for pixels, value := range training {
			network.BackProp(makeVec(pixels), makeOut(value), .2)
			//fmt.Println(makeOut(value))

		}
		fmt.Println("Epoch:", j)
		fmt.Println("Elapsed time:", time.Since(starttime))

	}

	for pixels, value := range training {

		fmt.Println(value)

		for i := 0; i < 28; i++ {
			for j := 0; j < 28; j++ {
				if pixels[28*i+j] > 50 {
					fmt.Print("X")
				} else {
					fmt.Print(" ")
				}

			}
			fmt.Print("\n")
		}

		ans, _ := network.Evaluate(makeVec(pixels))

		pretty(ans)
	}

	network.Save("pathhhh?")

}
