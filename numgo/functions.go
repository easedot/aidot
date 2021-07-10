package numgo

import (
	"gonum.org/v1/gonum/mat"
)

func reshape(x *Variable,s *Shape)*Variable{
	f:=NewFunction(&Reshape{S:s})
	y:=f.Run(x)
	return y
}
type Reshape struct {
	Function
	S *Shape
	Xs *Shape
}
func (s *Reshape) forward(ix[]*Variable)[]*Variable {
	x:=ix[0]
	s.Xs=x.Shape()
	return []*Variable{{Data:mat.NewDense(s.S.R, s.S.C, x.Data.RawMatrix().Data)}}
}
func (s *Reshape) backward(i,gy []*Variable)[]*Variable {
	return []*Variable{reshape(gy[0],s.Xs)}
}

func transpose(x *Variable)*Variable{
	f:=NewFunction(&Transpose{})
	y:=f.Run(x)
	return y
}
type Transpose struct {
	Function
}
func (t *Transpose) forward(ix[]*Variable)[]*Variable{
	x:=ix[0]
	return []*Variable{{Data:x.Data.T().(*mat.Dense)}}
}
func (t *Transpose) backward(i,gy []*Variable)[]*Variable{
	y:=gy[0]
	return []*Variable{transpose(y)}
}

func exp(x ...*Variable)*Variable{
	f:=NewFunction(&Exp{})
	return f.Run(x...)
}
type Exp struct {
	Function
}
func (e *Exp)forward(i []*Variable) []*Variable  {
	o:=mat.Dense{}
	o.Exp(i[0].Data)
	return [] *Variable{{Data:&o}}
}
func (e *Exp)backward(i, gy []*Variable) []*Variable {
	o:=mul(exp(i[0]), gy[0])
	return [] *Variable{o}
}

func sum(x *Variable)*Variable{
	f:=NewFunction(&Sum{})
	y:=f.Run(x)
	return y
}
type Sum struct {
	Function
	Xs *Shape
}
func (s *Sum) forward(ix[]*Variable)[]*Variable{
	x:=ix[0]
	s.Xs=x.Shape()
	y:=mat.Sum(x.Data)
	r:=NewVar(y)
	return []*Variable{r}
}
func (s *Sum) backward(i,gy []*Variable)[]*Variable{
	gx:=broadCastTo(gy[0],s.Xs)
	return []*Variable{gx}
}

func sumTo(x *Variable,s *Shape)*Variable{
	f:=NewFunction(&SumTo{S:s})
	y:=f.Run(x)
	return y
}
type SumTo struct {
	Function
	S *Shape
	Xs *Shape
}
func (s *SumTo) forward(ix[]*Variable)[]*Variable{
	x:=ix[0]
	s.Xs=x.Shape()
	y:= _sumTo(x,s.S)
	return []*Variable{y}
}
func (s *SumTo) backward(i,gy []*Variable)[]*Variable{
	g:=gy[0]
	gx:=broadCastTo(g,s.Xs)
	return []*Variable{gx}
}

func broadCastTo(x *Variable,s *Shape)*Variable{
	f:=NewFunction(&BroadcastTo{S:s})
	y:=f.Run(x)
	return y
}
type BroadcastTo struct {
	Function
	S *Shape
	Xs *Shape
}
func (s *BroadcastTo) forward(ix[]*Variable)[]*Variable{
	s.Xs =ix[0].Shape()
	y:=ix[0]
	if s.S.B(s.Xs){
		y=_broadcastTo(ix[0],s.S)
	}
	return []*Variable{y}
}
func (s *BroadcastTo) backward(i,gy []*Variable)[]*Variable{
	gx:=gy[0]
	if s.S.B(s.Xs){
		gx=sumTo(gy[0],s.Xs)
	}
	return []*Variable{gx}
}

func matmul(x,w *Variable)*Variable{
	f:=NewFunction(&BroadcastTo{})
	y:=f.Run(x)
	return y
}
type MatMul struct {
	Function
}
func (s *MatMul) forward(ix[]*Variable)[]*Variable{
	x,W:=ix[0],ix[1]
	y:=mat.Dense{}
	y.Mul(x.Data,W.Data)
	return []*Variable{{Data: &y}}
}
func (s *MatMul) backward(i,gy []*Variable)[]*Variable{
	x,W:=i[0],i[1]
	g:=gy[0]
	gx:=matmul(g,W.T())
	gW:=matmul(x.T(),g)
	return []*Variable{gx,gW}
}


