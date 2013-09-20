package ANN

import (
	"fmt"
	"math"
)

///aka "BPSig"
func bipolarSigmoid(in, sigma float64) float64 {
	temp := 1 + math.Exp(float64(-1.)*sigma*in)
	return 2./temp - 1
}

//would be nice to take advantage of the fact that this is in terms of
//original function
func bipolarSigmoidD(in, sigma float64) float64 {
	return .5 * (1 + bipolarSigmoid(in, sigma)) * (1 - bipolarSigmoid(in, sigma))
}

func identity(x float64) float64 {
	return x
}

func determineFunction(f func(float64) float64) string {
	err := .001

	variation := 0.
	for x := -1.; x < 1; x += .01 {
		variation += math.Abs(f(x) - bipolarSigmoid(x, 2))
	}
	if variation < err {
		return "BPSig (2)"
	}

	variation = 0.
	for x := -1.; x < 1; x += .01 {
		variation += math.Abs(f(x) - x)
	}
	if variation < err {
		return "Identity"
	}

	variation = 0.
	for x := -1.; x < 1; x += .01 {
		variation += math.Abs(f(x))
	}
	if variation < err {
		return "Zero"
	}

	variation = 0.
	for x := -1.; x < 1; x += .01 {
		variation += math.Abs(f(x) - 1)
	}
	if variation < err {
		return "One"
	}

	return "Unknown Function" + fmt.Sprint(variation)

}
