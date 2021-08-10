package tensor

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	ut "test_ai/utils"
)

func NewRand(shape ...int) *Tensor {
	s := Rand(shape...)
	return NewData(s, shape...)
}
func NewRandN(shape ...int) *Tensor {
	s := RandN(shape...)
	return NewData(s, shape...)
}
func NewRandE(shape ...int) *Tensor {
	s := RandE(shape...)
	return NewData(s, shape...)
}

func NewLinespace(start, stop float64, count int) *Tensor {
	s := ut.Linspace(start, stop, count)
	return NewData(s, len(s))
}

func NewArangeN(shape ...int) *Tensor {
	count := numElements(shape)
	return NewArange(0, float64(count), 1).View(shape...)
}

func NewArange(start, stop, step float64) *Tensor {
	s := ut.Arange(start, stop, step)
	t := NewData(s, len(s))
	return t
}

func NewOnes(shape ...int) *Tensor {
	d := make([]float64, numElements(shape))
	for di := range d {
		d[di] = 1
	}
	return NewData(d, shape...)
}
func LikeOnes(t *Tensor) *Tensor {
	return NewOnes(t.shape...)
}
func LikeZeros(t *Tensor) *Tensor {
	return NewZero(t.shape...)
}

func NewEyes(n int) *Tensor {
	d := NewZero(n, n)
	for i := 0; i < n; i++ {
		d.Set(1, i, i)
	}
	return d
}

func NewVar(v float64) *Tensor {
	d := []float64{v}
	return NewData(d, 1)
}

func NewVec(d ...float64) *Tensor {
	return NewData(d, len(d))
}

func NewData(v []float64, shape ...int) *Tensor {
	size := numElements(shape)
	if len(v) != size {
		panic(fmt.Errorf("data size must equal shape"))
	}
	t := &Tensor{
		ndim:    len(shape),
		shape:   shape,
		data:    v,
		Strides: make([]int, len(shape)),
	}
	initStrides(t)
	return t
}

func NewZero(shape ...int) *Tensor {
	size := numElements(shape)
	v := make([]float64, size)
	t := &Tensor{
		ndim:    len(shape),
		shape:   shape,
		data:    v,
		Strides: make([]int, len(shape)),
	}
	initStrides(t)
	return t
}

type Tensor struct {
	shape, Strides []int
	data           []float64
	ndim, offset   int
}

func (t *Tensor) Shape() []int {
	return t.shape
}

func (t *Tensor) Len() int {
	return numElements(t.shape)
}

func (t *Tensor) Print(name string) {
	s := t.Sprint(name)
	fmt.Print(s)
}
func (t *Tensor) Sprint(name string) string {
	s := fmt.Sprintf("%s\n%s\n", name, sprintTensor(t))
	return s
}
func (t *Tensor) Data() []float64 {
	return t.data
}
func (t *Tensor) Get(pos ...int) float64 {
	if len(pos) != t.ndim {
		panic(fmt.Errorf("pos must equal shape"))
	}
	index := t.offset + pos2index(pos, t.Strides)
	return t.data[index]
}

func (t *Tensor) Set(value float64, pos ...int) {
	if len(pos) != t.ndim {
		panic(fmt.Errorf("pos must equal shape"))
	}
	index := t.offset + pos2index(pos, t.Strides)
	t.data[index] = value
}

func (t *Tensor) Reshape(shape ...int) *Tensor {
	nt := t.Contingous().View(shape...)
	return nt
}

func (t *Tensor) Contingous() *Tensor {
	if isContiguous(t) {
		return t
	}
	nd := make([]float64, t.Len())
	nst := genStrides(t.shape...)
	for i := 0; i < t.Len(); i++ {
		pos := index2pos(i, nst)
		ncIndex := pos2index(pos, t.Strides)
		//fmt.Printf("p:%v i:%v\n", pos, ncIndex)
		nd[i] = t.data[t.offset+ncIndex]
	}
	nt := &Tensor{
		data:    nd,
		ndim:    t.ndim,
		shape:   t.shape,
		offset:  0,
		Strides: nst,
	}
	return nt
}

func (t *Tensor) View(shape ...int) *Tensor {
	if !isContiguous(t) {
		panic(fmt.Errorf("view tensor must contiguous"))
	}
	ndim := len(shape)
	nt := &Tensor{
		data:    t.data,
		offset:  t.offset,
		ndim:    ndim,
		shape:   make([]int, ndim),
		Strides: make([]int, ndim),
	}
	stride := 1
	for i := ndim - 1; i >= 0; i-- {
		nt.shape[i] = shape[i]
		nt.Strides[i] = stride
		stride *= shape[i]
	}
	return nt
}

func (t *Tensor) Index(idx, dim int) *Tensor {
	nt := &Tensor{
		data:    t.data,
		offset:  t.offset + idx*t.Strides[dim],
		ndim:    t.ndim - 1,
		shape:   make([]int, t.ndim-1),
		Strides: make([]int, t.ndim-1),
	}
	copy(nt.shape[0:], t.shape[:dim])
	copy(nt.shape[dim:], t.shape[dim+1:])

	copy(nt.Strides[0:], t.Strides[:dim])
	copy(nt.Strides[dim:], t.Strides[dim+1:])
	return nt
}

func (t *Tensor) Slice(start, end, dim int) *Tensor {
	nt := &Tensor{
		data:    t.data,
		offset:  t.offset + start*t.Strides[dim],
		ndim:    t.ndim,
		shape:   make([]int, t.ndim),
		Strides: make([]int, t.ndim),
	}
	copy(nt.shape, t.shape)
	copy(nt.Strides, t.Strides)
	nt.shape[dim] = end - start
	return nt
}

func Cat(dim int, tensors ...*Tensor) *Tensor {
	if len(tensors) <= 1 {
		panic(fmt.Errorf("cat input must >1"))
	}
	fistTensor := tensors[0]
	firstShape := fistTensor.shape
	dataSize := len(fistTensor.data)
	dimSize := fistTensor.shape[dim]
	for i := 1; i < len(tensors); i++ {
		tensor := tensors[i]
		dataSize += len(tensor.data)
		dimSize += tensor.shape[dim]
		if len(tensor.shape) != len(firstShape) {
			panic(fmt.Errorf("cat tensors must same ndim"))
		}
		for j := 0; j < len(firstShape); j++ {
			if j != dim && firstShape[j] != tensor.shape[j] {
				panic(fmt.Errorf("cat tensors must same dim size"))
			}

		}
	}
	newData := make([]float64, dataSize)
	nshape := make([]int, len(firstShape))
	copy(nshape, firstShape)
	nshape[dim] = dimSize
	nstrides := genStrides(nshape...)
	if dim == 0 {
		for i, ts := range tensors {
			dl := len(ts.data)
			copy(newData[i*dl:(i+1)*dl], ts.data[ts.offset:])
		}
	} else {
		preStrid := nstrides[dim-1]
		for i := 0; i < nshape[dim-1]; i++ {
			preOffset := i * preStrid
			for _, ts := range tensors {
				tsStrid := ts.Strides[dim-1]
				tsStart := ts.offset + i*tsStrid
				tsv := ts.data[tsStart : tsStart+tsStrid]
				copy(newData[preOffset:preOffset+tsStrid], tsv)
				preOffset += tsStrid
			}
		}
	}

	return NewData(newData, nshape...)
}

func (t *Tensor) Transpose(dim0, dim1 int) *Tensor {
	nt := &Tensor{
		data: t.data, offset: t.offset, ndim: t.ndim,
		shape:   make([]int, t.ndim),
		Strides: make([]int, t.ndim),
	}
	copy(nt.shape, t.shape)
	copy(nt.Strides, t.Strides)
	nt.shape[dim0], nt.shape[dim1] = t.shape[dim1], t.shape[dim0]
	nt.Strides[dim0], nt.Strides[dim1] = t.Strides[dim1], t.Strides[dim0]
	return nt
}

func (t *Tensor) T() *Tensor {
	if t.ndim != 2 {
		panic(fmt.Errorf("Tensor must 2 "))
	}
	return t.Transpose(0, 1)
}

func (t *Tensor) Permute(newIndex ...int) *Tensor {
	if len(newIndex) != len(t.shape) {
		panic(fmt.Errorf("Permute index ndim must equal old "))
	}
	nt := &Tensor{
		data: t.data, offset: t.offset, ndim: t.ndim,
		shape:   make([]int, t.ndim),
		Strides: make([]int, t.ndim),
	}
	for i, idx := range newIndex {
		nt.shape[i] = t.shape[idx]
		nt.Strides[i] = t.Strides[idx]
	}
	return nt
}

func (t *Tensor) Squeeze() *Tensor {
	ns := []int{}
	for i := 0; i < t.ndim; i++ {
		if t.shape[i] != 1 {
			ns = append(ns, t.shape[i])
		}
	}
	t.shape = ns
	t.Strides = genStrides(ns...)
	return t.View(ns...)
}

func (t *Tensor) UnSqueeze(dim int) *Tensor {
	t.shape = append(t.shape, 0)
	copy(t.shape[dim+1:], t.shape[dim:])
	t.shape[dim] = 1
	t.Strides = genStrides(t.shape...)
	return t.View(t.shape...)
}

func Equal(x, y *Tensor) bool {
	if !testEqInt(x.shape, y.shape) {
		return false
	}
	if !testEqInt(x.Strides, y.Strides) {
		return false
	}
	return true
}
func Eq(xi, yi *Tensor, e ...float64) bool {
	x, y := xi.Contingous(), yi.Contingous()
	if !testEqInt(x.shape, y.shape) {
		return false
	}
	if !testEqInt(x.Strides, y.Strides) {
		return false
	}
	//must continue
	if !testEq(x.data[x.offset:x.offset+x.Len()], y.data[y.offset:y.offset+y.Len()], e...) {
		return false
	}
	return true
}

func (t *Tensor) Clone() *Tensor {
	nt := LikeZeros(t)
	copy(nt.data, t.data)
	return nt
}

func Sin(t *Tensor) *Tensor {
	nt := t.Clone()
	nt.Sin()
	return nt
}

func (t *Tensor) Sin() {
	t.Apply(func(v float64) float64 {
		return math.Sin(v)
	})
}

func Cos(t *Tensor) *Tensor {
	nt := t.Clone()
	nt.Cos()
	return nt
}

func (t *Tensor) Cos() {
	t.Apply(func(v float64) float64 {
		return math.Cos(v)
	})
}

func Tanh(t *Tensor) *Tensor {
	nt := t.Clone()
	nt.Tanh()
	return nt
}

func (t *Tensor) Tanh() {
	t.Apply(func(v float64) float64 {
		return math.Tanh(v)
	})
}

func Pow(t *Tensor, n int) *Tensor {
	nt := t.Clone()
	nt.Pow(n)
	return nt
}

func (t *Tensor) Pow(n int) {
	t.Apply(func(v float64) float64 {
		return math.Pow(v, float64(n))
	})
}

func Exp(t *Tensor) *Tensor {
	nt := t.Clone()
	nt.Exp()
	return nt
}

func (t *Tensor) Exp() {
	t.Apply(func(v float64) float64 {
		return math.Exp(v)
	})
}
func Sqrt(t *Tensor) *Tensor {
	nt := t.Clone()
	nt.Sqrt()
	return nt
}

func (t *Tensor) Sqrt() {
	t.Apply(func(v float64) float64 {
		return math.Sqrt(v)
	})
}

func Neg(t *Tensor) *Tensor {
	nt := t.Clone()
	nt.Neg()
	return nt
}

func (t *Tensor) Neg() {
	t.Apply(func(v float64) float64 {
		return -v
	})
}

func Log(t *Tensor) *Tensor {
	nt := t.Clone()
	nt.Log()
	return nt
}

func (t *Tensor) Log() {
	t.Apply(func(v float64) float64 {
		return math.Log(v)
	})
}

func Clip(t *Tensor, min, max float64) *Tensor {
	nt := t.Clone()
	nt.Clip(min, max)
	return nt
}

func (t *Tensor) Clip(min, max float64) {
	t.Apply(func(v float64) float64 {
		if v > max {
			return max
		}
		if v < min {
			return min
		}
		return v
	})
}
func Maximum(t *Tensor, max float64) *Tensor {
	nt := t.Clone()
	nt.Maximum(max)
	return nt
}

func (t *Tensor) Maximum(max float64) {
	t.Apply(func(v float64) float64 {
		if v > max {
			return max
		}
		return v
	})
}
func Minimum(t *Tensor, min float64) *Tensor {
	nt := t.Clone()
	nt.Minimum(min)
	return nt
}

func (t *Tensor) Minimum(min float64) {
	t.Apply(func(v float64) float64 {
		if v > min {
			return min
		}
		return v
	})
}

func (t *Tensor) Apply(fn func(v float64) float64) {
	size := numElements(t.shape)
	for i := 0; i < size; i++ {
		t.data[i] = fn(t.data[i])
	}
}

func (t *Tensor) ApplyPos(fn func(pos []int, v float64) float64) {
	size := numElements(t.shape)
	for i := 0; i < size; i++ {
		pos := index2pos(i, t.Strides)
		t.data[i] = fn(pos, t.data[i])
	}
}

func (t *Tensor) ApplyPair(other *Tensor, fn func(v, ov float64) float64) {
	size := numElements(t.shape)
	for i := 0; i < size; i++ {
		t.data[i] = fn(t.data[i], other.data[i])
	}
}

//---tools

func testEq(a, b []float64, e ...float64) bool {
	if len(a) != len(b) {
		return false
	}
	eps := 1e-4
	if len(e) > 0 {
		eps = e[0]
	}
	for i := range a {
		if math.Abs(a[i]-b[i]) > eps {
			return false
		}
	}
	return true
}
func testEqInt(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func remove(slice []int, i int) []int {
	copy(slice[i:], slice[i+1:])
	return slice[:len(slice)-1]
}

func genStrides(shape ...int) []int {
	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}
	return strides
}
func initStrides(t *Tensor) {
	t.Strides = genStrides(t.shape...)
}

func isContiguous(t *Tensor) bool {
	sum := 1
	for i := len(t.shape) - 1; i >= 0; i-- {
		if t.Strides[i] != sum {
			return false
		}
		sum *= t.shape[i]
	}
	return true
}

func numElements(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}
func subElements(shape []int, start, end int) int {
	n := 1
	for ; start < end; start++ {
		n *= shape[start]
	}
	return n
}

func pos2index(pos []int, strides []int) int {
	index := 0
	for i := 0; i < len(strides); i++ {
		index += strides[i] * pos[i]
	}
	return index
}

func index2pos(index int, strides []int) []int {
	pos := make([]int, len(strides))
	for i := 0; i < len(strides); i++ {
		pos[i] = index / strides[i]
		index = index % strides[i]
	}
	return pos
}
func isBoradcastable(x, y []int) bool {
	xndim := len(x)
	yndim := len(y)
	if xndim < 1 || yndim < 1 {
		panic(fmt.Errorf("broadcast ndim must>=1"))
	}
	if xndim > yndim {
		paddingDim := xndim - yndim
		y = append(make([]int, paddingDim, paddingDim), y...)
	}
	if xndim < yndim {
		paddingDim := xndim - yndim
		x = append(x, make([]int, paddingDim)...)
	}
	for i := len(x) - 1; i >= 0; i-- {
		xdv := x[i]
		ydv := y[i]
		if xdv != ydv && !(xdv == 1 || ydv == 1) {
			return false
		}
	}
	return true
}

func sprintTensor(xi *Tensor) string {
	x := xi.Contingous()
	//x := xi
	sp := x.Shape()
	//先打最后核心维度
	s := ""
	num := numElements(sp)
	for i := 0; i < num; i++ {
		for j := 0; j < x.ndim-1; j++ {
			ds := i % x.Strides[j]
			if ds == 0 { //start
				if j == x.ndim-2 { //last dim
					s += fmt.Sprintf("%*s", j, " ")
					s += fmt.Sprintf("%s", "[")
				} else {
					s += fmt.Sprintf("%*s", j+1, "[")
				}
			}
		}

		s += fmt.Sprintf(" %.6f ", x.data[(x.offset+i)])

		for j := x.ndim - 2; j >= 0; j-- {
			ds := i % x.Strides[j]
			if ds == x.Strides[j]-1 { //end
				if j == x.ndim-2 { //last dim
					s += fmt.Sprintf("%s\n", "]")
				} else {
					s += fmt.Sprintf("\n%*s", j+1, "]")
				}
			}
		}
	}
	return s
}

func Rand(shape ...int) []float64 {
	size := numElements(shape)
	//ra:=rand.New(rand.NewSource(0))
	ra := rand.New(rand.NewSource(time.Now().UnixNano()))
	s := make([]float64, size, size)
	for i := range s {
		s[i] = ra.Float64()
	}
	return s
}
func RandN(shape ...int) []float64 {
	size := numElements(shape)
	//ra:=rand.New(rand.NewSource(0))
	ra := rand.New(rand.NewSource(time.Now().UnixNano()))
	s := make([]float64, size, size)
	for i := range s {
		s[i] = ra.NormFloat64()
	}
	return s
}
func RandE(shape ...int) []float64 {
	size := numElements(shape)
	//ra:=rand.New(rand.NewSource(0))
	ra := rand.New(rand.NewSource(time.Now().UnixNano()))
	s := make([]float64, size, size)
	for i := range s {
		s[i] = ra.ExpFloat64()
	}
	return s
}
