package numgo

import (
	nd "test_ai/numed"
)
func NewCompose(f...nd.EachFunc)*Compose{
	t:=&Compose{transforms: f}
	return t
}
type Compose struct {
	transforms [] nd.EachFunc
}
func (c Compose) Run(x[]float64) {
	xt:=nd.NewVec(x...)
	for _,t:=range c.transforms{
		if t!=nil{
			xt.Apply(t)
		}
	}
}

