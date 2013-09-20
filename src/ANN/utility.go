package ANN

import (
	"fmt" //testing
	"math/rand"
)

//Return a fully connected feedforward network with the given
//number of nodes, each of which use given activation function.
//Weights are all set to zero. Each layer contains a bias
func MakeSimple(inNodes, hidNodes, outNodes int, activation string) *Network {
	if activation != "BPSig" {
		panic("unsupported activation function")
	}

	sigma := 1.
	activate := func(in float64) float64 {
		return bipolarSigmoid(in, sigma)
	}
	activateD := func(in float64) float64 {
		return bipolarSigmoidD(in, sigma)
	}

	result := new(Network)
	result.inputNodes = make([]*node, inNodes+1)
	result.hiddenNodes = make([][]*node, 1)
	result.hiddenNodes[0] = make([]*node, hidNodes+1)
	result.outputNodes = make([]*node, outNodes)

	for i := 0; i < outNodes; i++ {
		newNode := new(node)
		newNode.activation = activate
		newNode.activationD = activateD
		result.outputNodes[i] = newNode
	}

	for i := 0; i < hidNodes; i++ {
		newNode := new(node)
		newNode.activation = activate
		newNode.activationD = activateD
		newNode.weights = make([]float64, outNodes)
		newNode.forward = make([]*node, outNodes)
		for j, out := range result.outputNodes {
			newNode.forward[j] = out
		}
		result.hiddenNodes[0][i] = newNode
	}

	bias := new(node)
	bias.activation = identity
	bias.activationD = identity
	bias.weights = make([]float64, outNodes)
	bias.forward = make([]*node, outNodes)
	for j, out := range result.outputNodes {
		bias.forward[j] = out
	}
	bias.state = 1

	result.hiddenNodes[0][hidNodes] = bias

	for i := 0; i < inNodes; i++ {
		newNode := new(node)
		newNode.activation = identity
		newNode.weights = make([]float64, hidNodes)
		newNode.forward = make([]*node, hidNodes)
		for j := 0; j < hidNodes; j++ {
			newNode.forward[j] = result.hiddenNodes[0][j]
		}
		result.inputNodes[i] = newNode
	}

	bias = new(node)
	bias.activation = identity
	bias.weights = make([]float64, hidNodes)
	bias.forward = make([]*node, hidNodes)
	for j := 0; j < hidNodes; j++ {
		bias.forward[j] = result.hiddenNodes[0][j]
	}
	bias.state = 1

	result.inputNodes[inNodes] = bias

	return result

}

//Testing purposes
func (a *Network) Print() {
	fmt.Println(a.outputFormat())
}

func (a *Network) Evaluate(inVector []float64) (answer []float64, err error) {
	if len(inVector)+1 != len(a.inputNodes) {
		return nil, &ErrWrongInput{len(a.inputNodes), len(inVector)}
	}
	a.zero()

	for i, input := range inVector {
		node := a.inputNodes[i]
		node.state = input
		signal := node.activation(node.state)
		for j, pass := range node.forward {
			pass.state += signal * node.weights[j]
		}
	}

	for _, layer := range a.hiddenNodes {
		for _, node := range layer {

			signal := node.activation(node.state)
			for j, pass := range node.forward {

				pass.state += signal * node.weights[j]
			}

		}

	}

	answer = make([]float64, len(a.outputNodes))

	for i, node := range a.outputNodes {
		answer[i] = node.activation(node.state)
	}

	return
}

func (a *Network) zero() {
	for _, layer := range a.hiddenNodes {
		for _, node := range layer {
			node.state = 0
		}
		//Bias
		layer[len(layer)-2].state = 1
	}
	for _, node := range a.outputNodes {
		node.state = 0
	}
}

//Sets all weights in network randomly in specified range
func (a *Network) RandomWeights(min, max float64) {
	width := max - min

	for _, node := range a.inputNodes {
		for j, _ := range node.weights {
			node.weights[j] = rand.Float64()*width + min
		}
	}

	for _, layer := range a.hiddenNodes {
		for _, node := range layer {
			for j, _ := range node.weights {
				node.weights[j] = rand.Float64()*width + min
			}
		}
	}

}
