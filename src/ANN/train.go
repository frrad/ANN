package ANN

import (
	"fmt"
)

func (a *Network) BackProp(in, target []float64, alpha float64) error {
	out, err := a.Evaluate(in)
	if err != nil {
		return err
	}

	for i, node := range a.outputNodes {
		fmt.Println("delta k=", (target[i]-out[i])*node.activationD(node.state))
	}

	return nil
}
