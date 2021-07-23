package numed

import (
	"fmt"
	logger "log"
	"math"
	"strings"

	"gonum.org/v1/gonum/blas/blas64"
	ut "test_ai/utils"
)



type Error struct{ string }
func (err Error) Error() string { return err.string }
type Float float64

var(
	ErrRowAccess           = Error{"mat: row index out of range"}
	ErrColAccess           = Error{"mat: column index out of range"}
	ErrColLength           = Error{"mat: col length mismatch"}
)

func NewShape(r,c int)*Shape{
	return &Shape{r,c}
}
type Shape struct {
	R,C int
}
type EachFunc func(i, j int, v float64) float64

func (s *Shape) E(i *Shape) bool{
	return s.R==i.R && s.C==i.C
}
func (s *Shape) G(i *Shape) bool{
	return s.R>i.R && s.C>=i.C||s.C>i.C && s.R>=i.R
}
func (s *Shape) L(i *Shape) bool{
	return s.R<i.R && s.C<=i.C||s.C<i.C && s.R<=i.R
}
func (s *Shape) X(i *Shape) bool{
	return s.R<i.R && s.C>i.C||s.C<i.C && s.R>i.R
}
func (s *Shape) B(i *Shape) bool{
	return s.BA(i)||s.BR(i)||s.BC(i)
}
func (s *Shape) BA(i *Shape) bool{
	//return s.R%i.R==0 && s.C%i.C==0
	return i.R==1 && i.C==1
}

func (s *Shape) BR(i *Shape) bool{
	return s.R%i.R==0 && s.C==i.C
}
func (s *Shape) BC(i *Shape) bool{
	return s.C%i.C==0 && s.R==i.R
}

func NewArange(start, stop, step float64)*NumEd{
	s:=ut.Arange(start,stop,step)
	return NewVec(s...)
}
func NewRand(r,c int) *NumEd {
	s:=ut.Rand(r,c)
	return NewMat(r,c,s...)
}
func NewRandN(r,c int) *NumEd {
	s:=ut.RandN(r,c)
	return NewMat(r,c,s...)
}
func NewRandE(r,c int) *NumEd {
	s:=ut.RandE(r,c)
	return NewMat(r,c,s...)
}
func NewVar(d float64)*NumEd{
	return NewMat(1,1,d)
}
func NewVec(d... float64)*NumEd{
	c:=len(d)
	return NewMat(1,c,d...)
}
func NewVecInt(d... int)*NumEd{
	c:=len(d)
	t:=make([]float64,c)
	for i,v:=range d{
		t[i]=float64(v)
	}
	return NewMat(1,c,t...)
}
func NewMat(r,c int,d...float64) *NumEd{
	if d==nil{
		d=make([]float64,r*c)
	}
	if len(d)<r*c{
		panic(ErrColLength)
	}
	ed:=&NumEd{blas64.General{Rows: r,Cols: c,Stride: c,Data: d}}
	return ed
}
func NewEyes(n int) *NumEd{
	d:= NewMat(n,n)
	for i:=0;i<n;i++{
		d.Set(i,i,1)
	}
	return d
}
func NewOnes(r,c int) *NumEd {
	d := make([]float64, r*c)
	for di:= range d{
		d[di]=1
	}
	return NewMat(r, c, d...)
}
func LikeOnes(n *NumEd) *NumEd {
	r,c:=n.Dims()
	return NewOnes(r,c)
}
func LikeZeros(n *NumEd) *NumEd {
	r,c:=n.Dims()
	return NewMat(r,c)
}

func Add(x,y *NumEd)*NumEd{
	x,y=_checkBroadCast(x,y)
	r,c:=x.Dims()
	size:=r*c
	t:=NewMat(r,c)
	for i:=0;i<size;i++{
		t.data.Data[i]=x.data.Data[i]+y.data.Data[i]
	}
	return t
}
func Sub(x,y *NumEd)*NumEd{
	x,y=_checkBroadCast(x,y)
	r,c:=x.Dims()
	size:=r*c
	t:=NewMat(r,c)
	for i:=0;i<size;i++{
		t.data.Data[i]=x.data.Data[i]-y.data.Data[i]
	}
	return t
}
func Mul(x,y *NumEd)*NumEd{
	x,y=_checkBroadCast(x,y)
	r,c:=x.Dims()
	size:=r*c
	t:=NewMat(r,c)
	for i:=0;i<size;i++{
		t.data.Data[i]=x.data.Data[i]*y.data.Data[i]
	}
	return t
}
func Div(x,y *NumEd)*NumEd{
	x,y=_checkBroadCast(x,y)
	r,c:=x.Dims()
	size:=r*c
	t:=NewMat(r,c)
	for i:=0;i<size;i++{
		t.data.Data[i]=x.data.Data[i]/y.data.Data[i]
	}
	return t
}
func Dot(x,y *NumEd)*NumEd{
	xr,_:=x.Dims()
	_,yc:=y.Dims()
	t:=NewMat(xr,yc)
	for r:=0;r<xr;r++{
		for c:=0;c<yc;c++{
			v:=Mul(x.Rows(r),y.Cols(c)).Sum(nil,true)
			t.set(r,c,v.Var())
		}
	}
	return t
}

func Cross(x,y *NumEd)*NumEd{
	xr,_:=x.Dims()
	t:=NewMat(xr,2)
	for i:=0;i<xr;i++{
		t.Set(i,0,x.Get(0,i))
		t.Set(i,1,y.Get(0,i))
	}
	return t
}
func MeshGrid(x,y *NumEd)(*NumEd,*NumEd){
	_,xc:=x.Dims()
	_,yc:=y.Dims()
	sp:=NewShape(yc,xc)
	y=y.T()
	xm:=broadcastTo(x,sp)
	ym:=broadcastTo(y,sp)
	return xm,ym
}
func Equal(x,y *NumEd) bool{
	if x==y{
		return true
	}
	xs,ys:=x.Shape(),y.Shape()
	eps:=1e-4
	if xs.E(ys){
		for i:=0;i<xs.R;i++{
			for j:=0;j<xs.C;j++{
				xv,yv:=x.get(i,j),y.get(i,j)
				if math.Abs(xv-yv)>eps{
					return false
				}
			}
		}
		return true
	}
	return false
}

type NumEd struct {
	data blas64.General
}
func (n *NumEd) Shape() *Shape {
	t:=NewShape(n.Dims())
	return t
}
func (n *NumEd) Dims() (r, c int) {
	return n.data.Rows,n.data.Cols
}
func (n *NumEd) Var() float64{
	return n.Get(0,0)
}
func (n *NumEd) IsVar() bool{
	return n.data.Rows==1 && n.data.Cols==1
}
func (n *NumEd) IsVec() bool{
	return n.data.Rows==1 && n.data.Cols!=1
}
func (n *NumEd) Set(i, j int, v float64) {
	n.checkRowCol(i, j)
	n.set(i, j, v)
}
func (n *NumEd) Get(i, j int) float64 {
	n.checkRowCol(i, j)
	return n.get(i, j)
}
//todo for use mat format
func (n *NumEd) At(i, j int) float64 {
	n.checkRowCol(i, j)
	return n.get(i, j)
}
func (n *NumEd) Apply(fn EachFunc) {
	ar, ac := n.Dims()
	for r := 0; r < ar; r++ {
		for c := 0; c < ac; c++ {
			n.set(r, c, fn(r, c, n.get(r, c)))
		}
	}
}
// Rows 间隔选取整行
func (n *NumEd) Rows (rs...int)*NumEd{
	mr:=len(rs)
	_,xc:=n.Dims()
	m:=NewMat(mr,xc)
	for i,r :=range rs{
		f:= n.Slice(r, r+1, 0, xc)
		t:= m.Slice(i, i+1,0,xc)
		t.Copy(f)
	}
	return m
}
// Rows 依次选取行，然后选取行中指定的列
func (n *NumEd) RowsCol (rcs ...int)*NumEd{
	mr:=len(rcs)
	m:=NewMat(mr,1)
	for i,c :=range rcs {
		f:= n.Slice(i, i+1,c,c+1)
		t:= m.Slice(i, i+1,c,c+1)
		t.Copy(f)
	}
	return m
}
// Cows 间隔选取整列
func (n *NumEd) Cols (cs ...int)*NumEd{
	mc:=len(cs)
	xr,_:=n.Dims()
	m:=NewMat(xr,mc)
	for i, c :=range cs {
		f:= n.Slice(0,xr,c, c+1)
		t:= m.Slice(0,xr,i, i+1)
		t.Copy(f)
	}
	return m
}
// CowsRow 依次选取列，然后选取列中指定的行
func (n *NumEd) ColsRow (crs ...int)*NumEd{
	mc :=len(crs)
	m:=NewMat(1,mc)
	for i, r :=range crs {
		f:= n.Slice(r, r+1,i, i+1)
		t:= m.Slice(r, r+1,i, i+1)
		t.Copy(f)
	}
	return m
}
func (n *NumEd) T()*NumEd{
	return n.tranposeTo()
}
func (n *NumEd)Sprint(name string) string {
	s:=fmt.Sprintf("%s\n%s\n",name, sprintNumEd(n))
	return s
}

//todo 目前是1维，以后考虑多维
func (n *NumEd) Ravel()[]float64{
	return n.data.Data
}
func (n *NumEd) Copy(a *NumEd) (r, c int) {
	r, c = a.Dims()
	for i:=0;i<r;i++{
		for j:=0;j<c;j++{
			n.Set(i,j,a.Get(i,j))
		}
	}
	return
}
func (n *NumEd) Slice(i,j,k,l int) *NumEd{
	return n.slice(i, k, j, l)
}
func (n *NumEd) Max(axis interface{},keepDims bool)*NumEd{
	return _max(n,axis,keepDims)
}
func (n *NumEd) Min(axis interface{},keepDims bool)*NumEd{
	return _min(n,axis,keepDims)
}
func (n *NumEd) ArgMax(axis interface{},keepDims bool)*NumEd{
	return _argmax(n,axis,keepDims)
}
func (n *NumEd) Sum(axis interface{},keepDims bool)*NumEd{
	return _sum(n,axis,keepDims)
}
func (n *NumEd) Mask(cond EachFunc)*NumEd{
	return mask(n,cond)
}

//---------utils
func mask(x *NumEd,cond EachFunc)*NumEd{
	xr,xc:=x.Dims()
	t:=LikeZeros(x)
	for i:=0;i<xr;i++{
		for j:=0;j<xc;j++{
			xv := x.get(i, j)
			t.set(i,j,cond(i,j, xv))
		}
	}
	return t
}

func sprintNumEd(x *NumEd) string {
	xr,xc:=x.Dims()
	r:=make([]string,xr)
	c:=make([]string,xc)
	for i:=0;i<xr;i++{
		for j:=0;j<xc;j++{
			c[j]=fmt.Sprintf("%v",x.get(i,j))
		}
		r[i]=fmt.Sprintf(" %s",strings.Join(c," "))
	}
	return strings.Join(r,"\n")
}

func _checkBroadCast(x0 *NumEd, x1 *NumEd) (*NumEd, *NumEd) {
	x0s,x1s:=x0.Shape(),x1.Shape()
	if !x0s.E(x1s) {
		if x0s.B(x1s) {
			x1 = broadcastTo(x1, x0s)
		}
		if x1s.B(x0s) {
			x0 = broadcastTo(x0, x1s)
		}
	}
	return x0,x1
}

func _max(x *NumEd,axis interface{},keepDims bool)*NumEd{
	mapFunc :=func(x,y,iy *NumEd,r,c,idx int){
		xv,yv := x.Get(r, c),y.Get(r, c)
		if xv>yv{
			y.Set(r, c, xv)
		}
	}
	y,_ := _rowColSet(x, mapFunc, axis, keepDims)
	return y
}
func _argmax(x *NumEd,axis interface{},keepDims bool)*NumEd{
	mapFunc :=func(x,y,iy *NumEd,r,c,idx int){
		xv,yv := x.Get(r, c),y.Get(r, c)
		if xv>yv{
			y.Set(r, c, xv)
			iy.Set(r, c, float64(idx))
		}
	}
	_,iy := _rowColSet(x, mapFunc, axis, keepDims)
	return iy
}

func _min(x *NumEd,axis interface{},keepDims bool)*NumEd{
	mapFunc :=func(x,y,iy *NumEd,r,c,idx int){
		xv,yv := x.Get(r, c),y.Get(r, c)
		if xv<yv||yv==0{
			y.Set(r, c, xv)
		}
	}
	y,_ := _rowColSet(x, mapFunc, axis, keepDims)
	return y
}

func _sum(x *NumEd,axis interface{},keepDims bool)*NumEd{
	mapFunc :=func(x,y,iy *NumEd,r,c,idx int){
		xv,yv := x.Get(r, c),y.Get(r, c)
		y.Set(r, c, xv+yv)
	}
	y,_ := _rowColSet(x, mapFunc, axis, keepDims)
	return y
}

type RowColFunc func(x,y,iy *NumEd,r,c,idx int)
func _rowColSet(x *NumEd,f RowColFunc,axis interface{},keepDims bool)(*NumEd,*NumEd){
	xr,xc:=x.Dims()
	to,toIdx :=&NumEd{},&NumEd{}
	if axis==nil{
		toIdx = NewMat(1, xc)
		tot:=NewMat(1, xc)
		_eachRow(x, f, tot,toIdx)
		//tot input col each
		to,toIdx = NewMat(1, 1),NewMat(1, 1)
		_eachCol(tot, f, to,toIdx)
	}else if axis.(int)==0{
		to,toIdx = NewMat(1, xc),NewMat(1, xc)
		_eachRow(x, f, to,toIdx)
	}else if axis.(int)==1 {
		to,toIdx = NewMat(xr, 1),NewMat(xr, 1)
		_eachCol(x, f, to,toIdx)
	}
	return to,toIdx
}
func _eachCol(x *NumEd, f RowColFunc, y,iy *NumEd) {
	xr,xc:=x.Dims()
	for ic := 0; ic < xc; ic++ {
		w := x.Slice(0, xr, ic, ic+1)
		for r := 0; r < xr; r++ {
			f(w, y,iy, r, 0,ic)
		}
	}
}

func _eachRow(x *NumEd, f RowColFunc, y,iy *NumEd) {
	xr,xc:=x.Dims()
	for ir := 0; ir < xr; ir++ {
		w := x.Slice(ir, ir+1, 0, xc)
		for c := 0; c < xc; c++ {
			f(w, y,iy, 0, c,ir)
		}
	}
}

func (n *NumEd) slice(i,j,k,l int) *NumEd{
	t := *n
	t.data.Data = t.data.Data[i*t.data.Stride+j : (k-1)*t.data.Stride+l]
	t.data.Rows = k - i
	t.data.Cols = l - j
	return &t

}

func (n *NumEd) get(i, j int) float64 {
	return n.data.Data[i*n.data.Stride+j]
}
func (n *NumEd) set(i, j int, v float64) {
	n.data.Data[i*n.data.Stride+j] = v
}
func (n *NumEd) checkRowCol(i int, j int) {
	if uint(i) >= uint(n.data.Rows) {
		panic(ErrRowAccess)
	}
	if uint(j) >= uint(n.data.Cols) {
		panic(ErrColAccess)
	}
}

func (n *NumEd) tranposeTo()*NumEd{
	xr,xc:=n.Dims()
	t:=NewMat(xc,xr)
	for i:=0;i<xr;i++ {
		for j:=0;j<xc;j++{
			t.Set(j,i,n.Get(i,j))
		}
	}
	return t
}

func broadcastTo(x *NumEd,s *Shape) *NumEd{
	xs:=x.Shape()
	xr,xc:=x.Dims()
	y:=NewMat(s.R,s.C)
	if s.BA(xs){
		//col
		for ic:=0;ic<s.C;ic+=xc{
			w := y.Slice(0, xr, ic, ic+xc)
			w.Copy(x)
		}
		wc:=y.Slice(0,1,0,s.C)
		//row
		for ir:=0;ir<s.R;ir+=xr{
			w := y.Slice(ir, ir+xr, 0, s.C)
			w.Copy(wc)
		}
		return y
	} else if s.BR(xs){
		for ir:=0;ir<s.R;ir+=xr{
			w := y.Slice(ir, ir+xr, 0, xc)
			w.Copy(x)
		}
	} else if s.BC(xs){
		for ic:=0;ic<s.C;ic+=xc{
			w := y.Slice(0, xr, ic, ic+xc)
			w.Copy(x)
		}
	}
	return y
}


func asVar(v interface{}) *NumEd{
	switch v.(type) {
	case []float64:
		return NewVec(v.([]float64)...)
	case float64:
		return NewVar(v.(float64))
	case NumEd:
		t:=v.(NumEd)
		return &t
	case *NumEd:
		return v.(*NumEd)
	case int:
		return NewVar(float64(v.(int)))
	default:
		logger.Printf("input type error")
		return nil
	}
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
//	// T returns the Transpose of the Matrix. Whether T returns a copy of the
//	// underlying data is implementation dependent.
//	// This method may be implemented using the transposeFunc type, which
//	// provides an implicit matrix Transpose.
//	T() Matrix
//}
//
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
