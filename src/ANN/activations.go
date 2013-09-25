package ANN

import (
	"fmt"
	"math"
)

func byID(id string) func(float64) float64 {
	if id == "BPSig" {
		return bipolarSigmoid
	}

	if id == "Identity" {
		return identity
	}

	panic("UNKNOWN FUNCTION")

	return nil
}

//derivative by ID (of original function)
func byIDD(id string) func(float64) float64 {
	if id == "BPSig" {
		return bipolarSigmoidD
	}

	if id == "Identity" {
		return identity
	}

	panic("UNKNOWN FUNCTION")

	return nil
}

///aka "BPSig"
func bipolarSigmoid(in float64) float64 {
	sigma := 1.
	temp := 1 + math.Exp(float64(-1.)*sigma*in)
	return 2./temp - 1
}

//would be nice to take advantage of the fact that this is in terms of
//original function
func bipolarSigmoidD(in float64) float64 {
	return .5 * (1 + bipolarSigmoid(in)) * (1 - bipolarSigmoid(in))
}

func identity(x float64) float64 {
	return x
}

func determineFunction(f func(float64) float64) string {
	err := .001
	variation := 0.

	variation = 0.
	for x := -1.; x < 1; x += .01 {
		variation += math.Abs(f(x) - bipolarSigmoid(x))
	}
	if variation < err {
		return fmt.Sprintf("BPSig")

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
