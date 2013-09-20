package ANN

import "math"

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
