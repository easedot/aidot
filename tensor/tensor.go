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
		Strides: genStrides(shape),
	}
	return t
}

func NewZero(shape ...int) *Tensor {
	size := numElements(shape)
	v := make([]float64, size)
	t := &Tensor{
		ndim:    len(shape),
		shape:   shape,
		data:    v,
		Strides: genStrides(shape),
	}
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
	s := ""
	if name != "" {
		s = fmt.Sprintf("%s\n%s\n", name, sprintTensor(t))
	} else {
		s = fmt.Sprintf("%s\n", sprintTensor(t))
	}

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
	nst := genStrides(t.shape)
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
	shape = autoFillNegShape(t.shape, shape)
	ndim := len(shape)
	nt := &Tensor{
		data:    t.data,
		offset:  t.offset,
		ndim:    ndim,
		shape:   make([]int, ndim),
		Strides: make([]int, ndim),
	}
	copy(nt.shape, shape)
	nt.Strides = genStrides(shape)
	return nt
}

func (t *Tensor) SumTo(dim int, keepDim bool) *Tensor {
	var nt *Tensor
	t.ApplyDim(dim, keepDim, func(idx int, dimT *Tensor) {
		if nt == nil {
			nt = dimT
		} else {
			nt = Add(nt, dimT)
		}
	})
	return nt
}

func (t *Tensor) MeanTo(dim int, keepDim bool) *Tensor {
	nt := t.SumTo(dim, keepDim)
	nt = Div(nt, NewVar(float64(t.shape[dim])))
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
			nt = argMax(nt, dimT, idt, idx)
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
			nt = Max(nt, dimT)
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
			nt = Min(nt, dimT)
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

func (t *Tensor) SliceSel(dim int, keepDim bool, sel1, sel2 []int) *Tensor {
	if len(sel1) != len(sel2) {
		panic(fmt.Errorf("sel1 len must equal sel2"))
	}
	nt := &Tensor{
		offset:  0,
		ndim:    t.ndim,
		shape:   make([]int, t.ndim),
		Strides: make([]int, t.ndim),
	}
	if keepDim {
		copy(nt.shape, t.shape)
		nt.shape[dim] = len(sel1)
		nt.Strides = genStrides(nt.shape)
		nt.data = make([]float64, numElements(nt.shape))
	} else {
		nt.shape = []int{len(sel1)}
		nt.ndim = 1
		nt.Strides = genStrides(nt.shape)
		nt.data = make([]float64, numElements(nt.shape))
	}
	dimLen := t.Strides[dim]

	preStride, preShape, oldPreStride := 0, 1, 0
	if dim > 0 {
		preStride = nt.Strides[dim-1]
		preShape = nt.shape[dim-1]
		oldPreStride = t.Strides[dim-1]
	}
	for i := 0; i < preShape; i++ {
		preOffset := i * preStride
		opreOffset := i * oldPreStride
		for j, s := range sel1 {
			newFrom := j * t.Strides[dim+1]
			newTo := (j + 1) * t.Strides[dim+1]
			if keepDim {
				newFrom = preOffset + j*dimLen + sel2[j]*nt.Strides[dim+1]
				newTo = preOffset + j*dimLen + (sel2[j]+1)*nt.Strides[dim+1]
			}
			oldFrom := t.offset + opreOffset + s*dimLen + sel2[j]*t.Strides[dim+1]
			oldTo := t.offset + opreOffset + s*dimLen + (sel2[j]+1)*t.Strides[dim+1]
			copy(nt.data[newFrom:newTo], t.data[oldFrom:oldTo])
		}
	}
	return nt
}

func (t *Tensor) Slices(dim int, sel ...int) *Tensor {
	nt := &Tensor{
		offset:  0,
		ndim:    t.ndim,
		shape:   make([]int, t.ndim),
		Strides: make([]int, t.ndim),
	}
	copy(nt.shape, t.shape)
	nt.shape[dim] = len(sel)
	nt.Strides = genStrides(nt.shape)
	nt.data = make([]float64, numElements(nt.shape))
	dimLen := nt.Strides[dim]

	preStride, preShape, oldPreStride := 0, 1, 0
	if dim > 0 {
		preStride = nt.Strides[dim-1]
		preShape = nt.shape[dim-1]
		oldPreStride = t.Strides[dim-1]
	}
	for i := 0; i < preShape; i++ {
		preOffset := i * preStride
		opreOffset := i * oldPreStride
		for j, s := range sel {
			newFrom := preOffset + j*dimLen
			newTo := preOffset + (j+1)*dimLen
			oldFrom := t.offset + opreOffset + s*dimLen
			oldTo := t.offset + opreOffset + (s+1)*dimLen
			copy(nt.data[newFrom:newTo], t.data[oldFrom:oldTo])
		}
	}
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
	dataSize := numElements(firstShape) // len(fistTensor.data)
	dimSize := fistTensor.shape[dim]
	for i := 1; i < len(tensors); i++ {
		tensor := tensors[i]
		dataSize += numElements(tensor.shape) // len(tensor.data)
		dimSize += tensor.shape[dim]
		if len(tensor.shape) != len(firstShape) {
			panic(fmt.Errorf("cat tensors must same ndim"))
		}
		for j := 0; j < len(firstShape); j++ {
			//除了cat的这个维度外，其他维度都相同才能cat
			if j != dim && firstShape[j] != tensor.shape[j] {
				panic(fmt.Errorf("cat tensors must same dim size"))
			}
		}
	}
	nt := &Tensor{
		offset:  0,
		ndim:    fistTensor.ndim,
		shape:   make([]int, len(firstShape)),
		Strides: make([]int, fistTensor.ndim),
		data:    make([]float64, dataSize),
	}
	copy(nt.shape, firstShape)
	nt.shape[dim] = dimSize
	nt.Strides = genStrides(nt.shape)

	if dim == 0 {
		for i, ts := range tensors {
			dl := numElements(ts.shape) // len(ts.data)
			copy(nt.data[i*dl:(i+1)*dl], ts.data[ts.offset:])
		}
	} else {
		preStrid := nt.Strides[dim-1]
		preShape := nt.shape[dim-1]
		for i := 0; i < preShape; i++ {
			preOffset := i * preStrid
			for _, ts := range tensors {
				tsPreStrid := ts.Strides[dim-1]

				tsStart := ts.offset + i*tsPreStrid
				tsEnd := tsStart + tsPreStrid
				copy(nt.data[preOffset:preOffset+tsPreStrid], ts.data[tsStart:tsEnd])
				preOffset += tsPreStrid
			}
		}
	}
	return nt
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

func Squeeze(t *Tensor) *Tensor {
	var ns []int
	for i := 0; i < t.ndim; i++ {
		if t.shape[i] != 1 {
			ns = append(ns, t.shape[i])
		}
	}
	nt := &Tensor{
		data:    t.data,
		ndim:    len(ns),
		shape:   ns,
		Strides: genStrides(ns),
		offset:  t.offset,
	}
	return nt
}

func UnSqueeze(t *Tensor, dim int) *Tensor {
	nt := &Tensor{
		data:  t.data,
		shape: make([]int, len(t.shape)+1),
	}
	copy(nt.shape, t.shape)
	copy(nt.shape[dim+1:], nt.shape[dim:])
	nt.shape[dim] = 1
	nt.Strides = genStrides(nt.shape)
	nt.ndim = len(nt.shape)
	return nt
}

//func UnSqueeze(t *Tensor, dim int) *Tensor {
//	nt := &Tensor{
//		data:  t.data,
//		shape: make([]int, len(t.shape)+1),
//	}
//	t.shape = append(t.shape, 0)
//	copy(t.shape[dim+1:], t.shape[dim:])
//	t.shape[dim] = 1
//	t.Strides = genStrides(t.shape...)
//	t.ndim = len(t.shape)
//	return t.View(t.shape...)
//}

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

func (t *Tensor) CloneFrom(o *Tensor) {
	//todo 这里用拷贝还是复用数据
	t.data = o.data
	copy(t.shape, o.shape)
	copy(t.Strides, o.Strides)
	t.ndim = o.ndim
	t.offset = o.offset
}

//Add func
func Add(t1, t2 *Tensor) *Tensor {
	nt := ApplyPair(t1, t2, func(idx int, v, ov float64) float64 {
		return v + ov
	})
	return nt
}

//Sub func
func Sub(t1, t2 *Tensor) *Tensor {
	nt := ApplyPair(t1, t2, func(idx int, v, ov float64) float64 {
		return v - ov
	})
	return nt
}

//Mul func
func Mul(t1, t2 *Tensor) *Tensor {
	nt := ApplyPair(t1, t2, func(idx int, v, ov float64) float64 {
		return v * ov
	})
	return nt
}

//Dot func
func Dot(t1, t2 *Tensor) *Tensor {
	dim := 0
	if t1.ndim == 1 && t2.ndim == 1 { //for 1D vector
		return Sum(Mul(t1, t2))
	}
	if t1.shape[0] != t2.shape[1] {
		panic(fmt.Errorf("dot a rows must equal b cols"))
	}
	t1r, t2c := t1.shape[dim], t2.shape[dim+1]
	nt := NewZero(t1r, t2c)
	for i := 0; i < t1r; i++ {
		for j := 0; j < t2c; j++ {
			sum := Mul(t1.Index(i, dim), t2.Index(j, dim+1)).Sum()
			nt.Set(sum, i, j)
		}
	}
	return nt
}

//Div func
func Div(t1, t2 *Tensor) *Tensor {
	nt := ApplyPair(t1, t2, func(idx int, v, ov float64) float64 {
		return v / ov
	})
	return nt
}

//Max func
func Max(t1, t2 *Tensor) *Tensor {
	nt := ApplyPair(t1, t2, func(idx int, v, ov float64) float64 {
		if v < ov {
			return ov
		}
		return v
	})
	return nt
}

////Max todo for cuda
//func (t *Tensor) Max(ot *Tensor) {
//	t.ApplyPair(ot, func(idx int, v, ov float64) float64 {
//		if v < ov {
//			return ov
//		}
//		return v
//	})
//}

//Min func
func Min(t1, t2 *Tensor) *Tensor {
	nt := ApplyPair(t1, t2, func(idx int, v, ov float64) float64 {
		if v > ov {
			return ov
		}
		return v
	})
	return nt
}

////Max todo for cuda
//func (t *Tensor) Min(ot *Tensor) {
//	t.ApplyPair(ot, func(idx int, v, ov float64) float64 {
//		if v > ov {
//			return ov
//		}
//		return v
//	})
//}

//---
func Sum(t *Tensor) *Tensor {
	return NewVar(t.Sum())
}

func (t *Tensor) Sum() float64 {
	sum := 0.0
	t.Apply(func(v float64) float64 {
		sum += v
		return v
	})
	return sum
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

func ApplyPair(t, other *Tensor, fn func(idx int, v, ov float64) float64) *Tensor {
	if !isBraadcastable(t.shape, other.shape) {
		panic(fmt.Errorf("calc tensor shape must isBraadcastable"))
	}
	if t.ndim > other.ndim {
		other = padTensor(t.ndim-other.ndim, other)
	}
	if t.ndim < other.ndim {
		t = padTensor(other.ndim-t.ndim, t)
	}
	nt := newUnionTensor(t, other)
	for i := 0; i < len(nt.data); i++ {
		pos := index2pos(i, nt.shape, nt.Strides)
		v := t.Get(pos...)
		ov := other.Get(pos...)
		nt.Set(fn(i, v, ov), pos...)
	}
	return nt
}

//func (t *Tensor) ApplyPair(other *Tensor, fn func(idx int, v, ov float64) float64) {
//	if !isBraadcastable(t.shape, other.shape) {
//		panic(fmt.Errorf("calc tensor shape must isBraadcastable"))
//	}
//	//todo pading shape
//	padShape(t, other)
//	size := numElements(t.shape)
//	ySize := numElements(other.shape)
//	if ySize > size {
//		calcPaire(other, t, fn)
//		//todo 这里目前为了效率使用的浅拷贝，复用了底层的data
//		t.CloneFrom(other)
//	} else {
//		calcPaire(t, other, fn)
//	}
//
//}
//func calcPaire(x, y *Tensor, fn func(idx int, v, ov float64) float64) {
//	size := numElements(x.shape)
//	for i := 0; i < size; i++ {
//		//pos := index2pos(i, x.shape, y.Strides)
//		pos := index2pos(i, x.shape, x.Strides)
//		//fmt.Printf("pos:%v", pos)
//		v := x.Get(pos...)
//		ov := y.Get(pos...)
//		x.Set(fn(i, v, ov), pos...)
//	}
//}
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

func newUnionTensor(x, y *Tensor) *Tensor {
	ndim := len(x.shape)
	nshape := make([]int, ndim)
	for i := 0; i < ndim; i++ {
		nshape[i] = x.shape[i]
		if y.shape[i] > x.shape[i] {
			nshape[i] = y.shape[i]
		}
	}
	nt := &Tensor{
		data:    make([]float64, numElements(nshape)),
		ndim:    ndim,
		shape:   nshape,
		Strides: genStrides(nshape),
	}
	return nt
}

//func padShape(x, y *Tensor) {
//	pad := len(x.shape) - len(y.shape)
//	if pad > 0 {
//		y = padTensor(pad, y)
//	}
//	if pad < 0 {
//		x = padTensor(-1*pad, x)
//	}
//}

func padTensor(pad int, t *Tensor) *Tensor {
	for i := 0; i < pad; i++ {
		t = UnSqueeze(t, 0)
	}
	return t
}

//argMax todo for cuda
func argMax(t, ot, idt *Tensor, idm int) *Tensor {
	nt := ApplyPair(t, ot, func(idx int, v, ov float64) float64 {
		if v < ov {
			pos := index2pos(idx, idt.shape, idt.Strides)
			idt.Set(float64(idm), pos...)
			return ov
		}
		return v
	})
	return nt
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

func genStrides(shape []int) []int {
	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}
	return strides
}
func initStrides(t *Tensor) {
	t.Strides = genStrides(t.shape)
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
	if xndim < 1 && yndim < 1 {
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
func autoFillNegShape(oldShape, newShape []int) []int {
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
					s += fmt.Sprintf("%*s", j, "")
					s += fmt.Sprintf("%s", "[")
				} else {
					s += fmt.Sprintf("%*s", j+1, "[")
				}
			}
		}

		s += fmt.Sprintf(" %9.6f ", x.data[(x.offset+i)])

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
