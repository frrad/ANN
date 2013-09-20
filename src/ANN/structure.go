package ANN

type node struct {
	state      float32
	activation *func(float32) float32
	forward    []*node
	weights    []float32
	back       []*node
}
