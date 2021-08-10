package numed

//
//import (
//	"fmt"
//
//	ut "test_ai/utils"
//)
//
//func NewArangeTensor(start, stop, step float64) *Tensor {
//	s := ut.Arange(start, stop, step)
//	t := NewTensor(1, len(s))
//	t.data = s
//	return t
//}
//
//func NewTensor(shape ...int) *Tensor {
//	size := 1
//	for _, v := range shape {
//		size *= v
//	}
//	v := make([]float64, size)
//	t := &Tensor{
//		ndim:    len(shape),
//		shape:   shape,
//		data:    v,
//		Strides: make([]int, len(shape)),
//	}
//	initStrides(t)
//	return t
//}
//
//type Tensor struct {
//	shape, Strides []int
//	data           []float64
//	ndim, offset   int
//}
//
//func (t *Tensor) Shape() []int {
//	return t.shape
//}
//
//func (t *Tensor) Len() int {
//	return numElements(t.shape)
//}
//
//func (t *Tensor) Print(name string) {
//	s := t.Sprint(name)
//	fmt.Print(s)
//}
//func (t *Tensor) Sprint(name string) string {
//	s := fmt.Sprintf("%s\n%s\n", name, sprintTensor(t))
//	return s
//}
//func (t *Tensor) Data() []float64 {
//	return t.data
//}
//func (t *Tensor) Get(pos ...int) float64 {
//	if len(pos) != t.ndim {
//		panic(fmt.Errorf("pos must equal shape"))
//	}
//	index := t.offset + pos2index(pos, t.Strides)
//	return t.data[index]
//}
//
//func (t *Tensor) Set(value float64, pos ...int) {
//	if len(pos) != t.ndim {
//		panic(fmt.Errorf("pos must equal shape"))
//	}
//	index := t.offset + pos2index(pos, t.Strides)
//	t.data[index] = value
//}
//
//func (t *Tensor) Reshape(shape ...int) *Tensor {
//	nt := t.Contingous().View(shape...)
//	return nt
//}
//
//func (t *Tensor) Contingous() *Tensor {
//	if isContiguous(t) {
//		return t
//	}
//	nd := make([]float64, t.Len())
//	nst := genStrides(t.shape...)
//	for i := 0; i < t.Len(); i++ {
//		pos := index2pos(i, nst)
//		ncIndex := pos2index(pos, t.Strides)
//		//fmt.Printf("p:%v i:%v\n", pos, ncIndex)
//		nd[i] = t.data[ncIndex]
//	}
//	nt := &Tensor{
//		data:    nd,
//		ndim:    t.ndim,
//		shape:   t.shape,
//		offset:  t.offset,
//		Strides: nst,
//	}
//	return nt
//}
//
//func (t *Tensor) View(shape ...int) *Tensor {
//	if !isContiguous(t) {
//		panic(fmt.Errorf("view tensor must contiguous"))
//	}
//	ndim := len(shape)
//	nt := &Tensor{
//		data:    t.data,
//		offset:  t.offset,
//		ndim:    ndim,
//		shape:   make([]int, ndim),
//		Strides: make([]int, ndim),
//	}
//	stride := 1
//	for i := ndim - 1; i >= 0; i-- {
//		nt.shape[i] = shape[i]
//		nt.Strides[i] = stride
//		stride *= shape[i]
//	}
//	return nt
//}
//
//func (t *Tensor) Index(idx, dim int) *Tensor {
//	nt := &Tensor{
//		data:    t.data,
//		offset:  t.offset + idx*t.Strides[dim],
//		ndim:    t.ndim - 1,
//		shape:   make([]int, t.ndim-1),
//		Strides: make([]int, t.ndim-1),
//	}
//	copy(nt.shape, t.shape)
//	copy(nt.Strides, t.Strides)
//	nt.shape = remove(nt.shape, dim)
//	nt.Strides = remove(nt.Strides, dim)
//	return nt
//}
//
//func (t *Tensor) Slice(start, end, dim int) *Tensor {
//	nt := &Tensor{
//		data:    t.data,
//		offset:  t.offset + start*t.Strides[dim],
//		ndim:    t.ndim,
//		shape:   make([]int, t.ndim),
//		Strides: make([]int, t.ndim),
//	}
//	copy(nt.shape, t.shape)
//	copy(nt.Strides, t.Strides)
//	nt.shape[dim] = end - start
//	return nt
//}
//
//func (t *Tensor) Transpose(dim0, dim1 int) *Tensor {
//	nt := &Tensor{
//		data: t.data, offset: t.offset, ndim: t.ndim,
//		shape:   make([]int, t.ndim),
//		Strides: make([]int, t.ndim),
//	}
//	copy(nt.shape, t.shape)
//	copy(nt.Strides, t.Strides)
//	nt.shape[dim0], nt.shape[dim1] = t.shape[dim1], t.shape[dim0]
//	nt.Strides[dim0], nt.Strides[dim1] = t.Strides[dim1], t.Strides[dim0]
//	return nt
//}
//
//func (t *Tensor) T() *Tensor {
//	if t.ndim != 2 {
//		panic(fmt.Errorf("Tensor must 2 "))
//	}
//	return t.Transpose(0, 1)
//}
//
//func (t *Tensor) Permute(newIndex ...int) *Tensor {
//	if len(newIndex) != len(t.shape) {
//		panic(fmt.Errorf("Permute index ndim must equal old "))
//	}
//	nt := &Tensor{
//		data: t.data, offset: t.offset, ndim: t.ndim,
//		shape:   make([]int, t.ndim),
//		Strides: make([]int, t.ndim),
//	}
//	for i, idx := range newIndex {
//		nt.shape[i] = t.shape[idx]
//		nt.Strides[i] = t.Strides[idx]
//	}
//	return nt
//}
//
//func (t *Tensor) Squeeze(dim int) *Tensor {
//	ns := []int{}
//	for i := 0; i < t.ndim; i++ {
//		if t.shape[i] != 1 {
//			ns = append(ns, t.shape[i])
//		}
//	}
//	return t.View(ns...)
//}
//
//func (t *Tensor) UnSqueeze(dim int) *Tensor {
//	t.shape = append(t.shape, 0)
//	copy(t.shape[dim+1:], t.shape[dim:])
//	return t.View(t.shape...)
//}
//
//func remove(slice []int, i int) []int {
//	copy(slice[i:], slice[i+1:])
//	return slice[:len(slice)-1]
//}
//
//func genStrides(shape ...int) []int {
//	strides := make([]int, len(shape))
//	stride := 1
//	for i := len(shape) - 1; i >= 0; i-- {
//		strides[i] = stride
//		stride *= shape[i]
//	}
//	return strides
//}
//func initStrides(t *Tensor) {
//	t.Strides = genStrides(t.shape...)
//}
//
//func isContiguous(t *Tensor) bool {
//	sum := 1
//	for i := len(t.shape) - 1; i >= 0; i-- {
//		if t.Strides[i] != sum {
//			return false
//		}
//		sum *= t.shape[i]
//	}
//	return true
//}
//
//func numElements(shape []int) int {
//	n := 1
//	for _, d := range shape {
//		n *= d
//	}
//	return n
//}
//func subElements(shape []int, start, end int) int {
//	n := 1
//	for ; start < end; start++ {
//		n *= shape[start]
//	}
//	return n
//}
//
//func pos2index(pos []int, strides []int) int {
//	index := 0
//	for i := 0; i < len(strides); i++ {
//		index += strides[i] * pos[i]
//	}
//	return index
//}
//
//func index2pos(index int, strides []int) []int {
//	pos := make([]int, len(strides))
//	for i := 0; i < len(strides); i++ {
//		pos[i] = index / strides[i]
//		index = index % strides[i]
//	}
//	return pos
//}
//func isBoradcastable(x, y []int) bool {
//	xndim := len(x)
//	yndim := len(y)
//	if xndim < 1 || yndim < 1 {
//		panic(fmt.Errorf("broadcast ndim must>=1"))
//	}
//	if xndim > yndim {
//		paddingDim := xndim - yndim
//		y = append(make([]int, paddingDim, paddingDim), y...)
//	}
//	if xndim < yndim {
//		paddingDim := xndim - yndim
//		x = append(x, make([]int, paddingDim)...)
//	}
//	for i := len(x) - 1; i >= 0; i-- {
//		xdv := x[i]
//		ydv := y[i]
//		if xdv != ydv && !(xdv == 1 || ydv == 1) {
//			return false
//		}
//	}
//	return true
//}
//
//func sprintTensor(xi *Tensor) string {
//	x := xi.Contingous()
//	//x := xi
//	sp := x.Shape()
//	//先打最后核心维度
//	s := ""
//	num := numElements(sp)
//	fmt.Printf("size %d", num)
//	for i := 0; i < num; i++ {
//		for j := 0; j < x.ndim-1; j++ {
//			ds := i % x.Strides[j]
//			if ds == 0 { //start
//				if j == x.ndim-2 { //last dim
//					s += fmt.Sprintf("\n%*s", j, " ")
//					s += fmt.Sprintf("%s", "[")
//				} else {
//					s += fmt.Sprintf("\n%*s", j+1, "[")
//				}
//			}
//		}
//
//		s += fmt.Sprintf(" %.8f ", x.data[i])
//
//		for j := x.ndim - 2; j >= 0; j-- {
//			ds := i % x.Strides[j]
//			if ds == x.Strides[j]-1 { //end
//				if j == x.ndim-2 { //last dim
//					s += fmt.Sprintf("%s", "]")
//				} else {
//					s += fmt.Sprintf("\n%*s", j+1, "]")
//				}
//			}
//		}
//	}
//	return s
//}
//func sprintTensor0(x *Tensor) string {
//	sp := x.Shape()
//	//先打最后核心维度
//	s := ""
//	num := numElements(sp)
//	fmt.Printf("size %d", num)
//	for i := 0; i < num; i++ {
//		for j := 0; j < x.ndim-1; j++ {
//			ds := i % x.Strides[j]
//			if ds == 0 {
//				s += fmt.Sprintf("\n%s", "[")
//			}
//			//if ds == 0 {
//			//	s += fmt.Sprintf("\n%*s", j+1, "[")
//			//}
//		}
//		s += fmt.Sprintf(" %.8f ", x.data[i])
//		for j := 0; j < x.ndim-1; j++ {
//			ds := i % x.Strides[j]
//			if ds == x.Strides[j]-1 {
//				s += fmt.Sprintf("%s", "]")
//				//if j < x.ndim-2 {
//				//	s += fmt.Sprintf("\n%*s", j, "")
//				//}
//			}
//		}
//	}
//	return s
//}
