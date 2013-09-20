package main

import (
	"ANN"
	"fmt"
	"io/ioutil"
	"strconv"
	"strings"
	"time"
)

const (
	size          = 10000
	dim           = 28
	testfreq      = 5
	testsize      = size / testfreq
	trainsize     = size - testsize
	lRate         = .0005
	targetPercent = .9
)

func readData(path string) (train, test map[[dim * dim]int]int) {
	ans, err := ioutil.ReadFile(path)
	if err != nil {
		panic("ERROR READING FILE")
	}

	test, train = make(map[[dim * dim]int]int), make(map[[dim * dim]int]int)

	splat := strings.Split(string(ans), "\n")

	for X, data := range splat {
		if X > size {
			break
		}

		split := strings.Split(data, ",")

		cheat, _ := strconv.Atoi(split[0])

		pixels := [dim * dim]int{}

		for i := 1; i < len(split); i++ {
			convert, _ := strconv.Atoi(split[i])
			pixels[i-1] = convert
		}

		if X%testfreq == 0 {
			test[pixels] = cheat
		} else {
			train[pixels] = cheat
		}
	}

	return
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

func max(a []float64) (ans int) {

	max := -5.

	for i, wt := range a {
		if wt > max {
			ans = i
			max = wt
		}
	}

	return

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

func show(pixels []float64) {
	for i := 0; i < 28; i++ {
		for j := 0; j < 28; j++ {
			if pixels[28*i+j] > .3 {
				fmt.Print("X")
			} else {
				fmt.Print(" ")
			}

		}
		fmt.Print("\n")
	}
}

func main() {

	starttime := time.Now()

	training, testing := readData("../data/train.csv")
	fmt.Println("Read in training data")

	network := ANN.MakeSimple(dim*dim, 100, 10, "BPSig")
	network.RandomWeights(-.001, .001)

	right := 0.

	for j := 0; right < testsize*targetPercent; j++ {

		//train
		for pixels, value := range training {

			network.BackProp(makeVec(pixels), makeOut(value), lRate)

		}

		fmt.Println("Epoch:", j, "Complete.")
		fmt.Println("Elapsed time:", time.Since(starttime))

		//test
		right = 0
		for pixels, value := range testing {
			ans, _ := network.Evaluate(makeVec(pixels))
			if max(ans) == value {
				right++
			}
		}
		fmt.Printf("Got %d/%d = %2.2f%%\n", int(right), testsize, 100*right/float64(testsize))

	}

	for pixels, value := range testing {

		fmt.Println(value)

		ans, _ := network.Evaluate(makeVec(pixels))
		answer := max(ans)
		show(makeVec(pixels))
		fmt.Printf("Answer: %d; Network thinks: %d; Probability: %2.1f %%\n", value, answer, 50*(ans[answer]+1))
		pretty(ans)
	}

	network.Save("network.net")

}
