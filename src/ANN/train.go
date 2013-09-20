package ANN

import (
//"fmt"
)

func (a *Network) BackProp(in, target []float64, alpha float64) error {
	if len(target) != len(a.outputNodes) {
		return &ErrWrongInput{len(a.outputNodes), len(target)}
	}

	out, err := a.Evaluate(in)
	if err != nil {
		return err
	}

	for i, node := range a.outputNodes {
		node.delta = (target[i] - out[i]) * node.activationD(node.state)
	}

	for _, layer := range a.hiddenNodes {
		for _, node := range layer {
			deltain := 0.
			for i, upnode := range node.forward {
				deltain += node.weights[i] * upnode.delta
			}
			node.delta = deltain * node.activationD(node.state)
		}
	}

	for _, inNode := range a.inputNodes {
		for i, link := range inNode.forward {
			update := alpha * link.delta * inNode.state
			inNode.weights[i] += update
		}
	}

	for _, layer := range a.hiddenNodes {
		for _, hidNode := range layer {
			for i, link := range hidNode.forward {
				update := alpha * link.delta * hidNode.state
				hidNode.weights[i] += update
			}
		}
	}

	return nil
}
