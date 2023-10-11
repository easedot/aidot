package numgo

import (
	nt "test_ai/tensor"
)

func Conv2d_sample(x, Wi, b *Variable, stride, pad int) *Variable {
	sp := x.Shape()
	N, _, H, W := sp[0], sp[1], sp[2], sp[3]
	wsp := Wi.Shape()
	OC, _, KH, KW := wsp[0], wsp[1], wsp[2], wsp[3]
	OH := getConvOutSize(H, KH, stride, pad)
	OW := getConvOutSize(W, KW, stride, pad)

	col := Img2col(x, KH, KW, stride, pad, true)
	wt := Wi.Reshape(OC, -1).Transpose(0, 1)
	//col(N*OH*OW,-1)
	//w(-1,OC)
	t := Linear(col, wt, b)
	y := t.Reshape(N, OH, OW, OC).Permute(0, 3, 1, 2)
	return y
}

func Pooling_simple(x *Variable, kh, kw, stride, pad int) *Variable {
	sp := x.Shape()
	N, C, H, W := sp[0], sp[1], sp[2], sp[3]

	OH := getConvOutSize(H, kh, stride, pad)
	OW := getConvOutSize(W, kw, stride, pad)
	col := Img2col(x, kh, kw, stride, pad, false)
	col = col.Reshape(-1, kh*kw)
	col = Max(col, false, 1)
	col = col.Reshape(N, OH, OW, C).Permute(0, 3, 1, 2)
	return col
}

type conv2d struct {
	Function
	stride, pad int
}

func (c *conv2d) forward(ix []*Variable) []*Variable {
	x, Wi, b := ix[0], ix[1], ix[2]
	sp := x.Shape()
	N, _, H, W := sp[0], sp[1], sp[2], sp[3]
	wsp := Wi.Shape()
	OC, _, KH, KW := wsp[0], wsp[1], wsp[2], wsp[3]
	OH := getConvOutSize(H, KH, c.stride, c.pad)
	OW := getConvOutSize(W, KW, c.stride, c.pad)

	col := Img2col(x, KH, KW, c.stride, c.pad, true)
	wt := Wi.Reshape(OC, -1).Transpose(0, 1)
	//col(N*OH*OW,-1)
	//w(-1,OC)
	t := Linear(col, wt, b)
	y := t.Reshape(N, OH, OW, OC).Permute(0, 3, 1, 2)
	return []*Variable{y}
}

func (c *conv2d) backward(ix, o, g []*Variable) []*Variable {
	gy := g[0]
	x, Wi := ix[0], ix[1]
	var b *Variable
	if len(ix) > 2 {
		b = ix[2]
	}

	wsp := Wi.Shape()
	FN, _, FH, FW := wsp[0], wsp[1], wsp[2], wsp[3]
	dout := gy.Permute(0, 2, 3, 1).Reshape(-1, FN)
	colW := Wi.Reshape(FN, -1).Transpose(0, 1)
	dcol := nt.Dot(dout.Data, colW.Data.T())
	dx := col2imArray(dcol, x.Shape(), FH, FW, c.stride, c.pad, true)
	gx := NewVariable(dx)
	db := &Variable{}
	if b != nil {
		db = NewVariable(nt.Sum(dout.Data, true, 0))
	}

	dw := nt.Dot(dout.Data, colW.Data.T())
	return []*Variable{gx, AsVar(dw), db}
}

//func (c *conv2d) backward(ix, o, g []*Variable) []*Variable {
//	gy := g[0]
//	x, Wi, b := ix[0], ix[1], ix[2]
//	oh, ow := x.Shape()[2], x.Shape()[3]
//	kh, kw := Wi.Shape()[2], Wi.Shape()[3]
//	gyv := []*Variable{gy}
//	gx := Deconv2d(gyv, oh, ow, c.stride, c.pad)
//
//	inp := []*Variable{x, gy}
//	gw := Conv2dGradW(inp, kh, kw, c.stride, c.pad)
//
//	gb := &Variable{}
//	if b.Data != nil {
//		gb = Sum(gy, false, 0, 2, 3)
//	}
//	return []*Variable{gx, gw, gb}
//}

func Conv2d(x []*Variable, stride, pad int) *Variable {
	f := NewFunction(&conv2d{pad: pad, stride: stride})
	return f.Run(x...)
}

//type deconv2d struct {
//	Function
//	stride, pad, oh, ow int
//}
//
//func (d *deconv2d) forward(ix []*Variable) []*Variable {
//	x, Wi, b := ix[0], ix[1], ix[2]
//	wsp := Wi.Shape()
//	xsp := x.Shape()
//	_, OC, KH, KW := wsp[0], wsp[1], wsp[2], wsp[3]
//	N, _, H, W := xsp[0], xsp[1], xsp[2], xsp[3]
//
//	if d.oh == 0 && d.ow == 0 {
//		d.oh = getConvOutSize(H, KH, d.stride, d.pad)
//		d.ow = getConvOutSize(W, KW, d.stride, d.pad)
//	}
//	imsp := []int{N, OC, d.oh, d.ow}
//
//	//gcol = xp.tensordot(Weight, x, (0, 1))
//	//gcol = np.transpose(gcol, (3, 0, 1, 2))
//
//	//gcol := Linear(x, Wi, b)
//
//	y := col2imArray(gcol.Data, imsp, KH, KW, d.stride, d.pad, false)
//	if b.Data != nil {
//		y = nt.Add(y, b.Data.Reshape(1, b.Data.Size(), 1, 1))
//	}
//	return []*Variable{NewVariable(y)}
//}
//
//func (d *deconv2d) backward(i, o, g []*Variable) []*Variable {
//	gy := g[0]
//	x, Wi, b := i[0], i[1], i[2]
//	kh, kw := Wi.Shape()[2], Wi.Shape()[3]
//	gx := Conv2d(i, d.stride, d.pad)
//	inp := []*Variable{gy, x}
//	gw := Conv2dGradW(inp, kh, kw, d.stride, d.pad)
//	gb := &nt.Tensor{}
//	if b.Data != nil {
//		gb = nt.Sum(gy.Data, false, 0, 2, 3)
//	}
//	gbv := NewVariable(gb)
//	return []*Variable{gx, gw, gbv}
//}
//
//func Deconv2d(x []*Variable, oh, ow, stride, pad int) *Variable {
//	f := NewFunction(&deconv2d{oh: oh, ow: ow, pad: pad, stride: stride})
//	return f.Run(x...)
//}
//
//type conv2dGradW struct {
//	Function
//	kh, kw, stride, pad int
//}
//
//func (d *conv2dGradW) forward(ix []*Variable) []*Variable {
//	x, gy := ix[0], ix[1]
//	col := im2colArray(x.Data, d.kh, d.kw, d.stride, d.pad, false)
//	//gW = xp.tensordot(gy, col, ((0, 2, 3), (0, 4, 5)))
//
//	return []*Variable{gw}
//}
//func (d *conv2dGradW) backward(i, o, g []*Variable) []*Variable {
//	x, gy := i[0], i[1]
//	gw := o[0]
//	xh, xw := x.Shape()[0], x.Shape()[1]
//	din := []*Variable{gy, gw}
//	gx := Deconv2d(din, xh, xw, d.stride, d.pad)
//
//	cin := []*Variable{x, gw}
//	ggy := Conv2d(cin, d.stride, d.pad)
//	return []*Variable{gx, ggy}
//}
//
//func Conv2dGradW(x []*Variable, kh, kw, stride, pad int) *Variable {
//	f := NewFunction(&conv2dGradW{kh: kh, kw: kw, pad: pad, stride: stride})
//	return f.Run(x...)
//}

type im2col struct {
	Function
	inputShape          []int
	kh, kw, stride, pad int
	toMatrix            bool
}

func (im *im2col) forward(ix []*Variable) []*Variable {
	x := ix[0]
	im.inputShape = x.Shape()
	y := im2colArray(x.Data, im.kh, im.kw, im.stride, im.pad, im.toMatrix)
	return []*Variable{NewVariable(y)}
}
func (im *im2col) backward(i, o, gy []*Variable) []*Variable {
	g := gy[0]
	gx := Col2im(g, im.inputShape, im.kh, im.kw, im.stride, im.pad, im.toMatrix)
	return []*Variable{gx}
}

func Img2col(x *Variable, kh, kw, stride, pad int, toMatrix bool) *Variable {
	f := NewFunction(&im2col{kh: kh, kw: kw, pad: pad, stride: stride, toMatrix: toMatrix})
	return f.Run(x)
}

type col2im struct {
	Function
	inputShape          []int
	kh, kw, stride, pad int
	toMatrix            bool
}

func (col *col2im) forward(ix []*Variable) []*Variable {
	x := ix[0]
	y := col2imArray(x.Data, col.inputShape, col.kh, col.kw, col.stride, col.pad, col.toMatrix)
	return []*Variable{NewVariable(y)}
}

func (col *col2im) backward(i, o, gy []*Variable) []*Variable {
	g := gy[0]
	gx := Img2col(g, col.kh, col.kw, col.stride, col.pad, col.toMatrix)

	return []*Variable{gx}
}

func Col2im(x *Variable, is []int, kh, kw, stride, pad int, toMatrix bool) *Variable {
	f := NewFunction(&col2im{inputShape: is, kh: kh, kw: kw, pad: pad, stride: stride, toMatrix: toMatrix})
	return f.Run(x)
}

// --
func getDeconvOutSize(size, k, s, p int) int {
	return s*(size-1) + k - 2*p
}
func getConvOutSize(inputSize, kernelSize, stride, pad int) int {
	return (inputSize+2*pad-kernelSize)/stride + 1
}

// data_t SRCNN::im2colGetPixel(data_t *im, ImageDim imageDim,
//
//	int row, int col, int channel, int pad)
//
//	{
//	   int height = get<1>(imageDim);
//	   int width = get<2>(imageDim);
//
//	   row -= pad;
//	   col -= pad;
//
//	   // zero padding
//
// #if 0
//
//	if(row < 0 || col < 0 || row >= height || col >= width)
//	{
//	    return 0;
//	}
//
// #endif
//
//	// reflect padding
//
// //#if 0
//
//	if(row < 0) row = 0;
//	if(col < 0) col = 0;
//	if(row >= height) row = height - 1;
//	if(col >= width) col = width - 1;
//
// //#endif
//
//	   return im[col + width * (row + height * channel)];
//	}
func im2ColGetPixel(img []float64, height, width, row, col, channel, pad int) float64 {
	row -= pad
	col -= pad
	if row < 0 || col < 0 || row >= height || col >= width {
		return 0
	}
	return img[col+width*(row+height*channel)]
}

//void SRCNN::col2imAddPixel(data_t *im, ImageDim imageDim,
//                           int row, int col, int channel, int pad, data_t value)
//{
//    int height = get<1>(imageDim);
//    int width = get<2>(imageDim);
//
//    row -= pad;
//    col -= pad;
//
//    // zero padding
//    if(row < 0 || col < 0 || row >= height || col >= width)
//    {
//        return;
//    }
//
//    im[col + width * (row + height * channel)] += value;
//}

func col2imAddPixel(img []float64, height, width, row, col, channel, pad int, value float64) bool {
	row -= pad
	col -= pad
	if row < 0 || col < 0 || row >= height || col >= width {
		return false
	}
	img[col+width*(row+height*channel)] += value
	return true
}

// void SRCNN::im2col(data_t *data_im, ImageDim imageDim, KernelDim kernelDim,
//
//	int stride, int pad, data_t *data_col)
//
//	{
//	   int imageHeight = get<1>(imageDim);
//	   int imageWidth = get<2>(imageDim);
//	   int kernelHeight = get<2>(kernelDim);
//	   int kernelWidth = get<3>(kernelDim);
//	   int col_height = (imageHeight + 2 * pad - kernelHeight) / stride + 1;
//	   int col_width = (imageWidth + 2 * pad - kernelWidth) / stride + 1;
//	   int imageChannel = get<0>(imageDim);
//	   int col_channel = imageChannel * kernelHeight * kernelWidth;
//
//	   for(int c = 0; c < col_channel; c++)
//	   {
//	       int w_offset = c % kernelWidth;
//	       int h_offset = (c / kernelWidth) % kernelHeight;
//	       int c_im = c / kernelWidth / kernelHeight;
//
// //#pragma omp parallel for
//
//	       for(int h = 0; h < col_height; h++)
//	       {
//	           for(int w = 0; w < col_width; w++)
//	           {
//	               int im_row = h_offset + h * stride;
//	               int im_col = w_offset + w * stride;
//	               int col_idx = (c * col_height + h) * col_width + w;
//	               data_col[col_idx] = im2colGetPixel(data_im, imageDim,
//	                                                  im_row, im_col, c_im, pad);
//	           }
//	       }
//	   }
//	}
func im2colArray(img *nt.Tensor, kh, kw, stride, pad int, toMatrix bool) *nt.Tensor {
	imgData := img.Data()

	sp := img.Shape()
	N, C, H, W := sp[0], sp[1], sp[2], sp[3]
	KH, KW := kh, kw
	OH := getConvOutSize(H, KH, stride, pad)
	OW := getConvOutSize(W, KW, stride, pad)
	//
	channelsCol := N * C * KH * KW
	var data = make([]float64, channelsCol*OH*OW)
	for c := 0; c < channelsCol; c++ {
		wOffset := c % kw
		hOffset := (c / kw) % kh
		cIm := c / kw / kh
		for h := 0; h < OH; h++ {
			for w := 0; w < OW; w++ {
				imRow := hOffset + h*stride
				imCol := wOffset + w*stride
				colIndex := (c*OH+h)*OW + w
				data[colIndex] = im2ColGetPixel(imgData, H, W, imRow, imCol, cIm, pad)
			}
		}
	}
	ret := nt.NewData(data, N, C, KH, KW, OH, OW)
	if toMatrix {
		ret = ret.Permute(0, 4, 5, 1, 2, 3).Reshape(N*OH*OW, -1)
	}
	return ret
}

// void SRCNN::col2im(data_t *data_col, ImageDim imageDim, KernelDim kernelDim,
//
//	int stride, int pad, data_t *data_im)
//
//	{
//	   int imageHeight = get<1>(imageDim);
//	   int imageWidth = get<2>(imageDim);
//	   int kernelHeight = get<2>(kernelDim);
//	   int kernelWidth = get<3>(kernelDim);
//	   int col_height = (imageHeight + 2 * pad - kernelHeight) / stride + 1;
//	   int col_width = (imageWidth + 2 * pad - kernelWidth) / stride + 1;
//	   int imageChannel = get<0>(imageDim);
//	   int col_channel = imageChannel * kernelHeight * kernelWidth;
//
//	   for(int c = 0; c < col_channel; c++)
//	   {
//	       int w_offset = c % kernelWidth;
//	       int h_offset = (c / kernelWidth) % kernelHeight;
//	       int c_im = c / kernelWidth / kernelHeight;
//
// //#pragma omp parallel for
//
//	       for(int h = 0; h < col_height; h++)
//	       {
//	           for(int w = 0; w < col_width; w++)
//	           {
//	               int im_row = h_offset + h * stride;
//	               int im_col = w_offset + w * stride;
//	               int col_idx = (c * col_height + h) * col_width + w;
//	               data_t value = data_col[col_idx];
//	               col2imAddPixel(data_im, imageDim, im_row, im_col,
//	                              c_im, pad, value);
//	           }
//	       }
//	   }
//	}
func col2imArray(col *nt.Tensor, imgShape []int, kh, kw, stride, pad int, toMatrix bool) *nt.Tensor {
	sp := imgShape
	N, C, H, W := sp[0], sp[1], sp[2], sp[3]
	KH, KW := kh, kw
	OH := getConvOutSize(H, KH, stride, pad)
	OW := getConvOutSize(W, KW, stride, pad)

	if toMatrix {
		col = col.Reshape(N, OH, OW, C, KH, KW).Permute(0, 3, 4, 5, 1, 2)
	}

	img := nt.NewZeros(N, C, H, W)
	channelsCol := N * C * KH * KW
	for c := 0; c < channelsCol; c++ {
		wOffset := c % kw
		hOffset := (c / kw) % kh
		cIm := c / kw / kh
		for h := 0; h < OH; h++ {
			for w := 0; w < OW; w++ {
				imRow := hOffset + h*stride
				imCol := wOffset + w*stride
				colIndex := (c*OH+h)*OW + w
				tv := col.Data()[colIndex]
				col2imAddPixel(img.Data(), H, W, imRow, imCol, cIm, pad, tv)
			}
		}
	}
	return img
}
