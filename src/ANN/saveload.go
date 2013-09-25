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

//Note to self: Check in to JSON
func ofParse(a string) *Network {
	lines := strings.Split(a, "\n")

	links := make(map[int]map[int]float64)
	layers := make([][]int, 1)
	layers[0] = make([]int, 0)
	function := make(map[int]string)
	currentLayer := 0

	tracker := -1

	for _, line := range lines {

		if len(line) == 0 {
			continue
		}

		if len(line) > 0 && string(line[0]) == "#" {
			continue
		}

		if len(line) >= 2 && line[:2] == "->" {
			dat := strings.Split(line[2:], " ")
			numba, err := strconv.Atoi(dat[0])
			functionID := dat[1]
			if err != nil {
				panic("TROUBLE PARSING NODE ID")
			}
			function[numba] = functionID

			if numba < tracker {
				currentLayer++
				layers = append(layers, make([]int, 0))
			}
			tracker = numba

			layers[currentLayer] = append(layers[currentLayer], numba)
		} else { //Line is assumed to be links
			source := layers[currentLayer][len(layers[currentLayer])-1]
			lList := strings.Split(line, "\t")
			links[source] = make(map[int]float64)
			for _, item := range lList {
				if len(item) == 0 {
					continue
				}
				item = strings.Replace(item, ":", "=", -1)
				datums := strings.Split(item, "=")
				from, err0 := strconv.Atoi(datums[0])
				to, err1 := strconv.Atoi(datums[1])
				weight, err2 := strconv.ParseFloat(datums[2], 64)
				if from != source {
					panic("weight / link mismatch")
				}
				if err0 != nil || err1 != nil || err2 != nil {
					panic("error converting from string")
				}
				links[from][to] = weight
			}

		}
	}

	// fmt.Println(layers, links, function)

	reverseLookup := make(map[int]*node)

	net := new(Network)

	net.inputNodes = make([]*node, len(layers[0]))
	for i := range net.inputNodes {
		node := new(node)
		id := layers[0][i]
		node.activation = byID(function[id])
		node.activationD = byIDD(function[id])
		// node.delta = float64(id)
		reverseLookup[layers[0][i]] = node
		net.inputNodes[i] = node
	}

	net.hiddenNodes = make([][]*node, 0)
	for hid := 1; hid <= len(layers)-2; hid++ {
		layer := make([]*node, len(layers[hid]))
		for i := 0; i < len(layer); i++ {
			node := new(node)
			id := layers[hid][i]
			node.activation = byID(function[id])
			node.activationD = byIDD(function[id])
			// node.delta = float64(id)
			reverseLookup[layers[hid][i]] = node
			layer[i] = node
		}
		net.hiddenNodes = append(net.hiddenNodes, layer)
	}

	lastPl := len(layers) - 1

	net.outputNodes = make([]*node, len(layers[lastPl]))
	for i := range net.outputNodes {
		node := new(node)
		id := layers[lastPl][i]
		node.activation = byID(function[id])
		node.activationD = byIDD(function[id])
		// node.delta = float64(id)
		reverseLookup[layers[lastPl][i]] = node
		net.outputNodes[i] = node
	}

	for from, chart := range links {
		for to, weight := range chart {
			fro := reverseLookup[from]
			ot := reverseLookup[to]
			fro.forward = append(fro.forward, ot)
			fro.weights = append(fro.weights, weight)
		}
	}

	return net

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
