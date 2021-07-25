package numgo

import (
	nd "test_ai/numed"
)

type UnitGrid struct{
	XX,YY,ZZ *nd.NumEd
}

func (g UnitGrid) Dims() (c, r int)   { r, c = g.XX.Dims(); return c, r }
func (g UnitGrid) Z(c, r int) float64 { return g.ZZ.At(r, c) }
func (g UnitGrid) X(c int) float64 {
	_, n := g.XX.Dims()
	if c < 0 || c >= n {
		panic("index out of range")
	}
	return g.XX.At(0,c)
}
func (g UnitGrid) Y(r int) float64 {
	m, _ := g.XX.Dims()
	if r < 0 || r >= m {
		panic("index out of range")
	}
	return g.YY.At(r,0)
}
