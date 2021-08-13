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
func NewRandNorm(shape ...int) *Tensor {
	s := RandN(shape...)
	return NewData(s, shape...)
}
func NewRandExp(shape ...int) *Tensor {
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
func (t *Tensor) Save(file string) {
	ut.WriteGob(file, t)
}
func (t *Tensor) Load(file string) {
	ut.ReadGob(file, &t)
}

func (t *Tensor) Data() []float64 {
	return t.data
}

func (t *Tensor) Get(pos ...int) float64 {
	if len(pos) != t.ndim {
		panic(fmt.Errorf("pos must equal shape"))
	}
	index := t.offset + pos2index(pos, t.shape, t.Strides)
	return t.data[index]
}

func (t *Tensor) Set(value float64, pos ...int) {
	if len(pos) != t.ndim {
		panic(fmt.Errorf("pos must equal shape"))
	}
	index := t.offset + pos2index(pos, t.shape, t.Strides)
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
		pos := index2pos(i, t.shape, nst)
		ncIndex := pos2index(pos, t.shape, t.Strides)
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
	shape = fillNegShape(t.shape, shape)
	ndim := len(shape)
	nt := &Tensor{
		data:    t.data,
		offset:  t.offset,
		ndim:    ndim,
		shape:   make([]int, ndim),
		Strides: make([]int, ndim),
	}
	copy(nt.shape, shape)
	nt.Strides = genStrides(shape...)
	return nt
}

func (t *Tensor) SumTo(dim int, keepDim bool) *Tensor {
	var nt *Tensor
	t.ApplyDim(dim, keepDim, func(idx int, dimT *Tensor) {
		if nt == nil {
			nt = dimT
		} else {
			nt.Add(dimT)
		}
	})
	return nt
}

func (t *Tensor) MeanTo(dim int, keepDim bool) *Tensor {
	nt := t.SumTo(dim, keepDim)
	nt.Div(NewVar(float64(t.shape[dim])))
	return nt
}

func (t *Tensor) ArgMax(dim int, keepDim bool) *Tensor {
	var nt *Tensor
	var idt *Tensor
	t.ApplyDim(dim, keepDim, func(idx int, dimT *Tensor) {
		if nt == nil {
			nt = dimT
			idt = LikeZeros(dimT)
		} else {
			nt.argMax(dimT, idt, idx)
		}
	})
	return idt
}

func (t *Tensor) MaxTo(dim int, keepDim bool) *Tensor {
	var nt *Tensor
	t.ApplyDim(dim, keepDim, func(idx int, dimT *Tensor) {
		if nt == nil {
			nt = dimT
		} else {
			nt.Max(dimT)
		}
	})
	return nt
}
func (t *Tensor) MinTo(dim int, keepDim bool) *Tensor {
	var nt *Tensor
	t.ApplyDim(dim, keepDim, func(idx int, dimT *Tensor) {
		if nt == nil {
			nt = dimT
		} else {
			nt.Min(dimT)
		}
	})
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

func (t *Tensor) SliceSelect(dim int, keepDim bool, sel ...int) *Tensor {
	ts := make([]*Tensor, len(sel))
	if keepDim {
		for i, s := range sel {
			ts[i] = t.Slice(s, s+1, dim)
		}
	} else {
		for i, s := range sel {
			ts[i] = t.Index(s, dim)
		}
	}
	return Cat(dim, ts...)
}

func (t *Tensor) Slices(dim int, sel ...int) *Tensor {
	nt := &Tensor{
		offset:  0,
		ndim:    t.ndim,
		shape:   make([]int, t.ndim),
		Strides: make([]int, t.ndim),
	}
	copy(nt.shape, t.shape)
	copy(nt.Strides, t.Strides)
	nt.shape[dim] = len(sel)
	nt.Strides = genStrides(nt.shape...)
	nt.data = make([]float64, numElements(nt.shape))
	dimLen := nt.Strides[dim]
	//if dim == 0 {
	//	for i, ts := range sel {
	//		copy(nt.data[i*dimLen:(i+1)*dimLen], t.data[t.offset+ts*dimLen:t.offset+(ts+1)*dimLen])
	//	}
	//} else {
	preStrid := 0
	opreStrid := 0
	preShape := 1
	if dim > 0 {
		preStrid = nt.Strides[dim-1]
		opreStrid = t.Strides[dim-1]
		preShape = nt.shape[dim-1]
	}
	for i := 0; i < preShape; i++ {
		preOffset := i * preStrid
		opreOffset := i * opreStrid
		for j, s := range sel {
			copy(
				nt.data[preOffset+j*dimLen:preOffset+(j+1)*dimLen],
				t.data[t.offset+opreOffset+s*dimLen:t.offset+opreOffset+(s+1)*dimLen])
		}
	}
	//}
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
	//todo 这里改了t也返回了新的tensor？
	t.shape = append(t.shape, 0)
	copy(t.shape[dim+1:], t.shape[dim:])
	t.shape[dim] = 1
	t.Strides = genStrides(t.shape...)
	t.ndim = len(t.shape)
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
	nt := &Tensor{
		offset:  t.offset,
		ndim:    t.ndim,
		shape:   make([]int, len(t.shape)),
		Strides: make([]int, len(t.Strides)),
		data:    make([]float64, len(t.data)),
	}
	copy(nt.data, t.data)
	copy(nt.shape, t.shape)
	copy(nt.Strides, t.Strides)
	return nt
}

//Add func
func Add(t1, t2 *Tensor) *Tensor {
	nt := t1.Clone()
	nt.Add(t2)
	return nt
}

//Add todo for cuda
func (t *Tensor) Add(ot *Tensor) {
	t.ApplyPair(ot, func(idx int, v, ov float64) float64 {
		return v + ov
	})
}

//Div func
func Div(t1, t2 *Tensor) *Tensor {
	nt := t1.Clone()
	nt.Add(t2)
	return nt
}

//Div todo for cuda
func (t *Tensor) Div(ot *Tensor) {
	t.ApplyPair(ot, func(idx int, v, ov float64) float64 {
		return v / ov
	})
}

//Max func
func Max(t1, t2 *Tensor) *Tensor {
	nt := t1.Clone()
	nt.Max(t2)
	return nt
}

//Max todo for cuda
func (t *Tensor) Max(ot *Tensor) {
	t.ApplyPair(ot, func(idx int, v, ov float64) float64 {
		if v < ov {
			return ov
		}
		return v
	})
}

//Min func
func Min(t1, t2 *Tensor) *Tensor {
	nt := t1.Clone()
	nt.Min(t2)
	return nt
}

//Max todo for cuda
func (t *Tensor) Min(ot *Tensor) {
	t.ApplyPair(ot, func(idx int, v, ov float64) float64 {
		if v > ov {
			return ov
		}
		return v
	})
}

//---

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
		t.data[i+t.offset] = fn(t.data[i+t.offset])
	}
}

func (t *Tensor) ApplyPos(fn func(pos []int, v float64) float64) {
	size := numElements(t.shape)
	for i := 0; i < size; i++ {
		pos := index2pos(i, t.shape, t.Strides)
		t.data[i+t.offset] = fn(pos, t.data[i+t.offset])
	}
}

func padShape(x, y *Tensor) {
	pad := len(x.shape) - len(y.shape)
	if pad > 0 {
		for i := 0; i < pad; i++ {
			//todo Unsqueeze 即修改了本身也返回了新的，这里先这样处理
			//y = y.UnSqueeze(0)
			y.UnSqueeze(0)
		}
	}
	if pad < 0 {
		for i := 0; i < -1*pad; i++ {
			//x = x.UnSqueeze(0)
			x.UnSqueeze(0)
		}
	}
}

func (t *Tensor) ApplyPair(other *Tensor, fn func(idx int, v, ov float64) float64) {
	if !isBraadcastable(t.shape, other.shape) {
		panic(fmt.Errorf("calc tensor shape must isBraadcastable"))
	}
	//todo pading shape
	padShape(t, other)
	size := numElements(t.shape)
	for i := 0; i < size; i++ {
		pos := index2pos(i, t.shape, t.Strides)
		//fmt.Printf("pos:%v", pos)
		v := t.Get(pos...)
		ov := other.Get(pos...)
		t.Set(fn(i, v, ov), pos...)
	}
}

func (t *Tensor) ApplyDim(dim int, keepDim bool, fn func(idx int, dimT *Tensor)) {
	dimSize := t.shape[dim]
	if keepDim {
		for i := 0; i < dimSize; i++ {
			fn(i, t.Slice(i, i+1, dim).Contingous())
		}
	} else {
		for i := 0; i < dimSize; i++ {
			fn(i, t.Index(i, dim).Contingous())
		}
	}
}

//---tools
//argMax todo for cuda
func (t *Tensor) argMax(ot, idt *Tensor, idm int) {
	t.ApplyPair(ot, func(idx int, v, ov float64) float64 {
		if v < ov {
			pos := index2pos(idx, idt.shape, idt.Strides)
			idt.Set(float64(idm), pos...)
			return ov
		}
		return v
	})
}

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

func pos2index(pos, shape, strides []int) int {
	index := 0
	for i := 0; i < len(strides); i++ {
		if shape[i] == 1 { //for broadcast
			index += 0
		} else {
			index += strides[i] * pos[i]
		}
	}
	return index
}

func index2pos(index int, shape, strides []int) []int {
	pos := make([]int, len(strides))
	for i := 0; i < len(strides); i++ {
		pos[i] = index / strides[i]
		index = index % strides[i]
	}
	return pos
}
func isBraadcastable(x, y []int) bool {
	xndim := len(x)
	yndim := len(y)
	if xndim < 1 || yndim < 1 {
		panic(fmt.Errorf("broadcast ndim must>=1"))
	}
	if xndim > yndim {
		paddingDim := xndim - yndim
		y = append(make([]int, paddingDim, paddingDim), y...)
		for i := 0; i < paddingDim; i++ {
			y[i] = 1
		}
	}
	if xndim < yndim {
		paddingDim := xndim - yndim
		x = append(x, make([]int, paddingDim, paddingDim)...)
		for i := 0; i < paddingDim; i++ {
			x[i] = 1
		}
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
func fillNegShape(oldShape, newShape []int) []int {
	oSize := numElements(oldShape)
	nSize, negPos := 1, -1
	for i, v := range newShape {
		if v == -1 {
			negPos = i
		} else {
			nSize *= v
		}
	}
	if negPos >= 0 {
		if oSize%nSize == 0 {
			newShape[negPos] = oSize / nSize
			return newShape
		} else {
			panic(fmt.Errorf("neg shape size not match old shape"))
		}
	} else {
		return newShape
	}
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
