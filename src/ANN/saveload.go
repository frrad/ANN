package ANN

import (
	"fmt"
	"io/ioutil"
)

func Load(path string) *Network {
	return nil //compile that fool
}

//Activation function is not saved!
func (a *Network) Save(path string) {
	ioutil.WriteFile(path, []byte(a.outputFormat()), 0666)
}

func (a *Network) outputFormat() (out string) {

	//Use delta to store node identifiers
	counter := 0.
	for _, node := range a.outputNodes {
		node.delta = counter
		counter++
	}

	for _, layer := range a.hiddenNodes {
		for _, node := range layer {
			node.delta = counter
			counter++
		}
	}
	for _, node := range a.inputNodes {
		node.delta = counter
		counter++
	}

	out += fmt.Sprint("#INPUT NODES")

	for _, node := range a.inputNodes {
		out += fmt.Sprint("\n->", node.delta, " ", determineFunction(node.activation), "\n")
		for i, target := range node.forward {
			out += fmt.Sprint(node.delta, ":", target.delta, "=", node.weights[i], "\t")
		}
	}

	out += fmt.Sprint("\n#HIDDEN NODES\n")

	for k, layer := range a.hiddenNodes {
		out += fmt.Sprint("#LAYER ", k)
		for _, node := range layer {
			out += fmt.Sprint("\n->", node.delta, " ", determineFunction(node.activation), "\n")
			for i, target := range node.forward {
				out += fmt.Sprint(node.delta, ":", target.delta, "=", node.weights[i], "\t")
			}
		}

	}
	out += fmt.Sprintln("\n#OUTPUT NODES")

	for _, node := range a.outputNodes {
		out += fmt.Sprint("->", node.delta, " ", determineFunction(node.activation), "\n")
	}

	return
}
