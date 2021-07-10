package numgo

import (
	"gonum.org/v1/gonum/mat"
	ut "test_ai/utils"
)
func pow(x *Variable,c int)*Variable{
	f:=NewFunction(&Pow{C:c})
	return f.Run(x)
}
type Pow struct {
	Function
	C int
}
func (s *Pow) forward(i []*Variable) []*Variable  {
	o:=mat.Dense{}
	o.Pow(i[0].Data,s.C)
	return []*Variable{{Data:&o}}
}

func (s *Pow) backward(i,gy []*Variable) []*Variable  {
	x:=i[0]
	c:=NewVar(float64(s.C))
	o:=mul(mul(pow(x,s.C-1),gy[0]),c)
	return []*Variable{o}
}

func neg(x *Variable)*Variable{
	f:=NewFunction(&Neg{})
	return f.Run(x)
}
type Neg struct {
	Function
}
func (e *Neg)forward(i []*Variable) []*Variable  {
	o:=mat.Dense{}
	o.Apply(ut.NegFunc,i[0].Data)
	return [] *Variable{{Data:&o}}
}
func (e *Neg)backward(i,gy []*Variable) []*Variable  {
	ngy:=neg(gy[0])
	return [] *Variable{ngy}
}

func add(x0,x1 *Variable)*Variable{
	f:=NewFunction(&Add{})
	y:=f.Run(x0,x1)
	return y
}
type Add struct {
	Function
	x0s *Shape
	x1s *Shape
}
func (a *Add) forward(ix []*Variable) []*Variable {
	x0,x1:=ix[0],ix[1]
	a.x0s, a.x1s =x0.Shape(),x1.Shape()
	x0, x1 = _checkBroadCast(a.x0s, a.x1s, x0, x1)
	o := mat.Dense{}
	o.Add(x0.Data, x1.Data)
	return []*Variable{{Data: &o}}
}
func (a *Add) backward(i,gy []*Variable) []*Variable  {
	gx0,gx1:=gy[0],gy[0]
	gx0, gx1 = _checkSumTo(a.x0s, a.x1s, gx0, gx1)
	return []*Variable{gx0, gx1}
}

func sub(x0,x1 *Variable)*Variable{
	f:=NewFunction(&Sub{})
	y:=f.Run(x0,x1)
	return y
}
type Sub struct {
	Function
	x0s *Shape
	x1s *Shape
}
func (a *Sub) forward(ix []*Variable) []*Variable {
	x0,x1:=ix[0],ix[1]
	a.x0s, a.x1s =x0.Shape(),x1.Shape()
	x0, x1 = _checkBroadCast(a.x0s, a.x1s, x0, x1)
	o:=mat.Dense{}
	o.Sub(x0.Data,x1.Data)
	return []*Variable{{Data:&o}}
}
func (a *Sub) backward(i,gy []*Variable) []*Variable  {
	g:=gy[0]
	gx0,gx1:=g,neg(g)
	gx0, gx1 = _checkSumTo(a.x0s, a.x1s, gx0, gx1)
	return []*Variable{gx0,gx1}
}


func mul(x0,x1 *Variable)*Variable{
	f:=NewFunction(&Mul{})
	y:=f.Run(x0,x1)
	return y
}
type Mul struct {
	Function
	x0s *Shape
	x1s *Shape
}
func (m *Mul) forward(ix[]*Variable)[]*Variable  {
	x0,x1:=ix[0],ix[1]
	m.x0s, m.x1s =x0.Shape(),x1.Shape()
	x0, x1 = _checkBroadCast(m.x0s, m.x1s, x0, x1)
	o:=mat.Dense{}
	o.MulElem(x0.Data,x1.Data)
	return []*Variable{{Data: &o}}
}
func (m *Mul) backward(i,gy []*Variable)[]*Variable  {
	g:=gy[0]
	x0,x1:=i[0],i[1]
	gx0:=mul(x1,g)
	gx1:=mul(x0,g)
	gx0,gx1  = _checkSumTo(m.x0s, m.x1s, gx0, gx1)
	return []*Variable{gx0,gx1}
}

func div(x0,x1 *Variable)*Variable {
	f:=NewFunction(&Div{})
	y:=f.Run(x0,x1)
	return y
}
type Div struct {
	Function
	x0s *Shape
	x1s *Shape
}
func (d *Div) forward(ix[]*Variable)[]*Variable {
	x0,x1:=ix[0],ix[1]
	d.x0s, d.x1s =x0.Shape(),x1.Shape()
	x0, x1 = _checkBroadCast(d.x0s, d.x1s, x0, x1)
	o:=mat.Dense{}
	o.DivElem(x0.Data,x1.Data)
	return []*Variable{{Data:&o}}
}

func (d *Div) backward(i,gy []*Variable)[]*Variable {
	g:=gy[0]
	x0,x1:=i[0],i[1]
	gx0 :=div(g,x1)
	gx1 :=mul(g, div(neg(x0), pow(x1, 2)))
	return []*Variable{gx0, gx1}
}




//import (
//	//"gonum.org/v1/gonum/blas/blas32"
//	"gonum.org/v1/gonum/blas/blas64"
//)
//type Error struct{ string }
//
//func (err Error) Error() string { return err.string }
//
//var (
//	ErrNegativeDimension   = Error{"mat: negative dimension"}
//	ErrIndexOutOfRange     = Error{"mat: index out of range"}
//	ErrReuseNonEmpty       = Error{"mat: reuse of non-empty matrix"}
//	ErrRowAccess           = Error{"mat: row index out of range"}
//	ErrColAccess           = Error{"mat: column index out of range"}
//	ErrVectorAccess        = Error{"mat: vector index out of range"}
//	ErrZeroLength          = Error{"mat: zero length in matrix dimension"}
//	ErrRowLength           = Error{"mat: row length mismatch"}
//	ErrColLength           = Error{"mat: col length mismatch"}
//	ErrSquare              = Error{"mat: expect square matrix"}
//	ErrNormOrder           = Error{"mat: invalid norm order for matrix"}
//	ErrSingular            = Error{"mat: matrix is singular"}
//	ErrShape               = Error{"mat: dimension mismatch"}
//	ErrIllegalStride       = Error{"mat: illegal stride"}
//	ErrPivot               = Error{"mat: malformed pivot list"}
//	ErrTriangle            = Error{"mat: triangular storage mismatch"}
//	ErrTriangleSet         = Error{"mat: triangular set out of bounds"}
//	ErrBandwidth           = Error{"mat: bandwidth out of range"}
//	ErrBandSet             = Error{"mat: band set out of bounds"}
//	ErrDiagSet             = Error{"mat: diagonal set out of bounds"}
//	ErrSliceLengthMismatch = Error{"mat: input slice length mismatch"}
//	ErrNotPSD              = Error{"mat: input not positive symmetric definite"}
//	ErrFailedEigen         = Error{"mat: eigendecomposition not successful"}
//)
//// ErrorStack represents matrix handling errors that have been recovered by Maybe wrappers.
//type ErrorStack struct {
//	Err error
//
//	// StackTrace is the stack trace
//	// recovered by Maybe, MaybeFloat
//	// or MaybeComplex.
//	StackTrace string
//}
//
//func (err ErrorStack) Error() string { return err.Err.Error() }
//
//const badCap = "mat: bad capacity"
//
//
//type Matrix interface {
//	// Dims returns the dimensions of a Matrix.
//	Dims() (r, c int)
//
//	// At returns the value of a matrix element at row i, column j.
//	// It will panic if i or j are out of bounds for the matrix.
//	At(i, j int) float64
//
//	// T returns the transpose of the Matrix. Whether T returns a copy of the
//	// underlying data is implementation dependent.
//	// This method may be implemented using the Transpose type, which
//	// provides an implicit matrix transpose.
//	T() Matrix
//}
//
//type Dense struct {
//	mat blas64.General
//
//	capRows, capCols int
//}
//
//func NewDense(r, c int, data []float64) *Dense {
//	if r <= 0 || c <= 0 {
//		if r == 0 || c == 0 {
//			panic(ErrZeroLength)
//		}
//		panic(ErrNegativeDimension)
//	}
//	if data != nil && r*c != len(data) {
//		panic(ErrShape)
//	}
//	if data == nil {
//		data = make([]float64, r*c)
//	}
//	return &Dense{
//		mat: blas64.General{
//			Rows:   r,
//			Cols:   c,
//			Stride: c,
//			Data:   data,
//		},
//		capRows: r,
//		capCols: c,
//	}
//}
//
//func(m *Dense) Dims() (r, c int){
//	return m.mat.Rows,m.mat.Cols
//}
//
//func (m *Dense) Mul(a, b Matrix) {
//
//}
//func (m *Dense) IsEmpty() bool {
//	// It must be the case that m.Dims() returns
//	// zeros in this case. See comment in Reset().
//	return m.mat.Stride == 0
//}
//
//func (m *Dense) Apply(fn func(i, j int, v float64) float64, a Matrix) {
//	ar, ac := a.Dims()
//	for r := 0; r < ar; r++ {
//		for c := 0; c < ac; c++ {
//			m.set(r, c, fn(r, c, a.At(r, c)))
//		}
//	}
//}
//// use returns a float64 slice with l elements, using f if it
//// has the necessary capacity, otherwise creating a new slice.
//func use(f []float64, l int) []float64 {
//	if l <= cap(f) {
//		return f[:l]
//	}
//	return make([]float64, l)
//}
//
//func (m *Dense) reuseAsNonZeroed(r, c int) {
//	// reuseAs must be kept in sync with reuseAsZeroed.
//	if m.mat.Rows > m.capRows || m.mat.Cols > m.capCols {
//		// Panic as a string, not a mat.Error.
//		panic(badCap)
//	}
//	if r == 0 || c == 0 {
//		panic(ErrZeroLength)
//	}
//	if m.IsEmpty() {
//		m.mat = blas64.General{
//			Rows:   r,
//			Cols:   c,
//			Stride: c,
//			Data:   use(m.mat.Data, r*c),
//		}
//		m.capRows = r
//		m.capCols = c
//		return
//	}
//	if r != m.mat.Rows || c != m.mat.Cols {
//		panic(ErrShape)
//	}
//}
//// At returns the element at row i, column j.
//func (m *Dense) At(i, j int) float64 {
//	if uint(i) >= uint(m.mat.Rows) {
//		panic(ErrRowAccess)
//	}
//	if uint(j) >= uint(m.mat.Cols) {
//		panic(ErrColAccess)
//	}
//	return m.at(i, j)
//}
//
//func (m *Dense) at(i, j int) float64 {
//	return m.mat.Data[i*m.mat.Stride+j]
//}
//
//// Set sets the element at row i, column j to the value v.
//func (m *Dense) Set(i, j int, v float64) {
//	if uint(i) >= uint(m.mat.Rows) {
//		panic(ErrRowAccess)
//	}
//	if uint(j) >= uint(m.mat.Cols) {
//		panic(ErrColAccess)
//	}
//	m.set(i, j, v)
//}
//
//func (m *Dense) set(i, j int, v float64) {
//	m.mat.Data[i*m.mat.Stride+j] = v
//}
