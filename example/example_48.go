package main

import "C"
import (
	"fmt"
	"image/color"
	"math/rand"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/palette"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	ng "test_ai/numgo"
	ut "test_ai/utils"
)

func main() {
	maxEpoch:=300
	batchSize:=30

	xd,t:=ng.GetSpiral(true)
	xm:=mat.NewDense(ut.Flatten(xd))

	lr:=1.0
	sgd:=ng.SGD(lr)
	//sgd:=ng.Adam(0.01,0.9,0.999,1e-8)
	model:=ng.MLP(ng.Sigmoid, sgd,  10, 3)

	dataSize:= len(xd)
	//maxIter:=int(math.Ceil(float64(dataSize)/float64(batchSize)))
	maxIter:=dataSize/batchSize

	for i:=0;i<maxEpoch;i++{
		sumLoss:=0.0
		//rand.Seed(time.Now().UnixNano())
		index:=rand.Perm(dataSize)
		//index:=ut.ArangeInt(0,dataSize,1)
		for j:=0;j<maxIter;j++{
			batchIndex:=index[j*batchSize:(j+1)*batchSize]
			batchX:=ng.SelRow(xm,batchIndex...)
			batchT:=ut.SelRowInt(t,batchIndex...)
			x := &ng.Variable{Data: batchX}
			y := model.Forward(x)
			loss :=ng.SoftmaxCrossEntroy(y,batchT)
			if i==0 && j==0{
				loss.Plot(true,"./temp/spiral.png")
			}
			model.ClearGrad()
			loss.Backward(false)
			model.Grad2Param()

			l := loss.At(0, 0)
			sumLoss+= l *float64(len(batchT))
		}
		avgLoss:=sumLoss/float64(dataSize)
		fmt.Printf("epoch %d loss:%4f\n",i+1,avgLoss)
	}

	x0:=ng.ColData(xm,0)
	xmin,xmax:=ut.MinMaxFloat64Slice(x0)
	x1:=ng.ColData(xm,1)
	ymin,ymax:=ut.MinMaxFloat64Slice(x1)
	h := 0.01
	vecx := ng.NewVec(ut.Arange(xmin, xmax, h)...)
	vecy := ng.NewVec(ut.Arange(ymin, ymax, h)...)
	xv,yv:=ng.MeshGrid(vecx, vecy)
	X:=ut.CrossSlice(ng.Ravel(xv.Data),ng.Ravel(yv.Data))
	XX:=mat.NewDense(ut.Flatten(X))
	score:=model.Forward(&ng.Variable{Data: XX})
	//score.Print("score")
	z:=ng.AgrMax(score,1,true)
	z=ng.Reshape(z,xv.Shape())
	//z.Print("Z")
	ug:=ng.UnitGrid{xv,yv,z}

	p := plot.New()
	p.Title.Text = "Plotutil example"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"
	p.Add(plotter.NewGrid())

	//predPts:=make(plotter.XYs, xr)
	pts1:=make(plotter.XYs,1)
	pts2:=make(plotter.XYs,1)
	pts3:=make(plotter.XYs,1)
	for i:=0;i< len(xd);i++{
		p:=plotter.XY{}
		p.X=xd[i][0]
		p.Y=xd[i][1]
		if t[i]==0{
			pts1=append(pts1,p)
		}
		if t[i]==1{
			pts2=append(pts2,p)
		}
		if t[i]==2{
			pts3=append(pts3,p)
		}
	}
	ptsList:=[]plotter.XYs{pts1,pts2,pts3}

	var pls []plot.Plotter
	c := plotter.NewContour(
		ug,
		nil,
		palette.Rainbow(10, palette.Blue, palette.Red, 1, 1, 1),
	)
	pls=append(pls,c)
	for _,sd:=range ptsList{
		s, err := plotter.NewScatter(sd)
		if err != nil {
			panic(err)
		}
		s.GlyphStyle.Color = color.RGBA{R: uint8(rand.Intn(255)), B: uint8(rand.Intn(255)), A: uint8(rand.Intn(255))}
		pls=append(pls,s)
	}


	p.Add(pls...)
	if err := p.Save(10*vg.Inch, 6*vg.Inch, "./temp/points.png"); err != nil {
		panic(err)
	}

}

