package ANN

import (
	"fmt"
)

type node struct {
	state       float64
	delta       float64
	activation  func(float64) float64
	activationD func(float64) float64
	forward     []*node
	weights     []float64
	//back       []*node
}

type Network struct {
	inputNodes  []*node
	hiddenNodes [][]*node //one slice per layer
	outputNodes []*node
}

type ErrWrongInput struct {
	expected, supplied int
}

func (e *ErrWrongInput) Error() string {
	return fmt.Sprintf("Expected %d inputs, got %d", e.expected, e.supplied)
}
