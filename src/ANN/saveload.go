package ANN

import (
	"fmt"
	"io/ioutil"
	"strconv"
	"strings"
)

func Load(path string) *Network {

	//note to self -- better error handling
	str, err := ioutil.ReadFile(path)
	if err != nil {
		panic("ERROR READING FILE")
	}

	toParse := string(str)

	net := ofParse(toParse)

	return net

}

//Activation function is not saved!
func (a *Network) Save(path string) {
	ioutil.WriteFile(path, []byte(a.outputFormat()), 0666)
}

func ofParse(a string) *Network {
	lines := strings.Split(a, "\n")

	links := make(map[int][]int)
	allnodes := make([]int, 0)

	for _, line := range lines {
		links[2] = []int{3}
		//fmt.Println(line)
		if line[:2] == "->" {
			dat := strings.Split(line[2:], " ")
			numba, err := strconv.Atoi(dat[0])
			functionID := dat[1]
			if err != nil {
				panic("TROUBLE PARSING NODE ID")
			}
			fmt.Println(line)
			fmt.Println(numba, functionID)
		}
	}

	return nil
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
