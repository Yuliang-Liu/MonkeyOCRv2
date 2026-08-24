#ifndef MAT2D_H_
#define MAT2D_H_

#include <vector>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <fstream>

namespace jpegio {

template <class T>
class mat2D
{
public:
	int rows;
	int cols;

private:
	std::vector<T> vect;

public:

    T* GetBuffer()
    {
        return &this->vect[0];
    }

	mat2D(int rows, int cols)
	{
		this->rows = rows;
		this->cols = cols;
		this->vect = std::vector<T>(rows*cols);
	}

	mat2D(int rows, int cols, T baseVal)
	{
		this->rows = rows;
		this->cols = cols;
		this->vect = std::vector<T>(rows*cols, baseVal);
	}

	~mat2D()
	{
		vect.clear();
	}

	T Read(int row, int col)
	{
		return this->vect[row*cols+col];
	}

	int valueCount(T val)
	{
		return std::count(this->vect.begin(), this->vect.end(), val);
	}

	void Write(int row, int col, T val)
	{
		this->vect[row*cols+col] = val;
	}

	void PermuteElements()
	{
		random_shuffle(this->vect.begin(), this->vect.end());
	}

	void PermuteRows()
	{
		std::vector<std::vector<T> > rowVect = std::vector<std::vector<T> >(this->rows, 0);
		for (int row=0; row < this->rows; row++)
		{
			std::vector<T> colVect = std::vector<T>(this->cols, 0);
			for (int col=0; col < this->cols; col++)
				colVect[col] = this->Read(row, col);
			rowVect[row] = colVect;
		}

		random_shuffle(rowVect.begin(), rowVect.end());

		for (int row=0; row < this->rows; row++)
		{
			for (int col=0; col < this->cols; col++)
				Write(row, col, rowVect[row][col]);
		}
		rowVect.clear();
	}

	void Print(int rowFrom, int rowTo, int colFrom, int colTo)
	{
		std::cout << "\n";
		for (int r=rowFrom; r<=rowTo; r++)
		{	
			for (int c=colFrom; c<=colTo; c++)
			{
				std::cout << this->Read(r, c) << " ";
			}
			std::cout << "\n";
		}
	}

	void Print()
	{
		Print(0, this->rows-1, 0, this->cols-1);
	}
	
	void PrintToFile(std::string full_path)
	{
		std::ofstream outputFile;
		outputFile.open(full_path.c_str());
		for (int row=0; row < this->rows; row++)
		{
			for (int col=0; col < this->cols; col++)
			{
				outputFile << this->Read(row, col) << " ";
			}
			outputFile << std::endl;
		}
		outputFile.close();
	}
	
	// -----------------------------------------------------
	//					STATIC METHODS
	// -----------------------------------------------------
	static mat2D<double>* ChangeToAbsValue(mat2D<double> *mat)
	{
		for (int r=0; r<mat->rows; r++)
			for (int c=0; c<mat->cols; c++)
				mat->Write(r, c, fabs(mat->Read(r, c)));
		return mat;
	}

	static mat2D<double>* Retype_int2double(mat2D<int> *mat)
	{
		mat2D<double> * result = new mat2D<double>(mat->rows, mat->cols);
		for (int rows=0; rows < mat->rows; rows++)
			for (int cols=0; cols < mat->cols; cols++)
				result->Write(rows, cols, (double)mat->Read(rows, cols));
		return result;
	}

	static mat2D<float>* Retype_int2float(mat2D<int> *mat)
	{
		mat2D<float> * result = new mat2D<float>(mat->rows, mat->cols);
		for (int rows=0; rows < mat->rows; rows++)
			for (int cols=0; cols < mat->cols; cols++)
				result->Write(rows, cols, (float)mat->Read(rows, cols));
		return result;
	}

	static mat2D<T>* Transpose(mat2D<T> *mat)
	{
		mat2D<T> * result = new mat2D<T>(mat->cols, mat->rows);
		for (int rows=0; rows < mat->rows; rows++)
			for (int cols=0; cols < mat->cols; cols++)
				result->Write(cols, rows, mat->Read(rows, cols));
		return result;
	}

	static mat2D<T>* Rotate180(mat2D<T> *mat)
	{
		mat2D<T> * result = new mat2D<T>(mat->rows, mat->cols);
		for (int row=0; row < mat->rows; row++)
			for (int col=0; col < mat->cols; col++)
				result->Write(mat->rows-row-1, mat->cols-col-1, mat->Read(row, col));
		return result;
	}

	static mat2D<T>* Submatrix(mat2D<T> *mat, int rowFrom, int rowTo, int colFrom, int colTo)
	{
		if ((rowFrom > rowTo) || (colFrom > colTo)) throw("Wrong submatrix input arguments.");
		mat2D<T> * result = new mat2D<T>(rowTo-rowFrom+1, colTo-colFrom+1);
		for (int r = rowFrom; r <= rowTo; r++)
			for (int c = colFrom; c <= colTo; c++)
				result->Write(r-rowFrom, c-colFrom, mat->Read(r, c));
		return result;
	}

	static mat2D<T>* AbsoluteValue(mat2D<T> *mat)
	{
		mat2D<T> * result = new mat2D<T>(mat->rows, mat->cols);
		for (int rows=0; rows < mat->rows; rows++)
			for (int cols=0; cols < mat->cols; cols++)
				result->Write(rows, cols, fabs(mat->Read(rows, cols)));
		return result;
	}

	static T SumElements(mat2D<T> *mat)
	{
		T result = 0;
		for (int row=0; row < mat->rows; row++)
			for (int col=0; col < mat->cols; col++)
				result += mat->Read(row, col);
		return result;
	}

	static mat2D<T> * AddMatrices(mat2D<T> * mat1, mat2D<T> * mat2)
	{
		if ((mat1->rows != mat2->rows) || (mat1->cols != mat2->cols))
			throw("Matrices must have same size.");
		mat2D<T> * result = new mat2D<T>(mat1->rows, mat1->cols);
		for (int row=0; row < mat1->rows; row++)
			for (int col=0; col < mat1->cols; col++)
				result->Write(row, col, mat1->Read(row, col) + mat2->Read(row, col));
		return result;
	}

	static mat2D<T> * AddValue(mat2D<T> * mat, T value)
	{
		mat2D<T> * result = new mat2D<T>(mat->rows, mat->cols);
		for (int row=0; row < mat->rows; row++)
			for (int col=0; col < mat->cols; col++)
				result->Write(row, col, mat->Read(row, col) + value);
		return result;
	}

	static mat2D<T> * SubtractMatrices(mat2D<T> * mat1, mat2D<T> * mat2)
	{
		if ((mat1->rows != mat2->rows) || (mat1->cols != mat2->cols))
			throw("Matrices must have same size.");
		mat2D<T> * result = new mat2D<T>(mat1->rows, mat1->cols);
		for (int row=0; row < mat1->rows; row++)
			for (int col=0; col < mat1->cols; col++)
				result->Write(row, col, mat1->Read(row, col) - mat2->Read(row, col));
		return result;
	}

	static mat2D<T> * MultiplyByNumber(mat2D<T> * mat, T number)
	{
		mat2D<T> * result = new mat2D<T>(mat->rows, mat->cols);
		for (int row=0; row < mat->rows; row++)
			for (int col=0; col < mat->cols; col++)
				result->Write(row, col, number * mat->Read(row, col));
		return result;
	}

	static mat2D<T> * MultiplyByMatrixElementwise(mat2D<T> * mat1, mat2D<T> * mat2)
	{
		if ((mat1->rows != mat2->rows) || (mat1->cols != mat2->cols))
			throw("Matrices must have same size.");
		mat2D<T> * result = new mat2D<T>(mat1->rows, mat1->cols);
		for (int row=0; row < mat1->rows; row++)
			for (int col=0; col < mat1->cols; col++)
				result->Write(row, col, mat1->Read(row, col) * mat2->Read(row, col));
		return result;
	}

	static mat2D<T> * MultiplyByMatrix(mat2D<T> * mat1, mat2D<T> * mat2)
	{
		if (mat1->cols != mat2->rows)
			return NULL;
		mat2D<T> * result = new mat2D<T>(mat1->rows, mat2->cols);
		for (int row=0; row < mat1->rows; row++)
		{
			for (int col=0; col < mat2->cols; col++)
			{
				T tempSum = 0;
				for (int mat1cols=0; mat1cols<mat1->cols; mat1cols++)
					for (int mat2rows=0; mat2rows<mat2->rows; mat2rows++)
						tempSum += mat1->Read(row, mat1cols) * mat2->Read(mat2rows, col);
				result->Write(row, col, tempSum);
			}
		}
		return result;
	}

	static mat2D<T> * InvertValues(mat2D<T> * mat)
	{
		mat2D<T> * result = new mat2D<T>(mat->rows, mat->cols);
		for (int row=0; row < mat->rows; row++)
			for (int col=0; col < mat->cols; col++)
				result->Write(row, col, ((T)1) / mat->Read(row, col));
		return result;
	}

	static mat2D<T> * PowerByElements(mat2D<T> *mat, double power)
	{
		mat2D<T> * result = new mat2D<T>(mat->rows, mat->cols);
		for (int row=0; row < mat->rows; row++)
			for (int col=0; col < mat->cols; col++)
				result->Write(row, col, pow(mat->Read(row, col), power));
		return result;
	}

	static mat2D<T> * Correlation_Same_basicFilters(mat2D<T> * mat, mat2D<T> * F1, mat2D<T> * F2)
	{
		mat2D<T> * temp1 = mat2D<T>::Correlation_Same(mat, F1);
		mat2D<T> * result = mat2D<T>::Correlation_Same(temp1, F2);
		delete temp1;

		return result;
	}

	static mat2D<T> * Convolution_Same_basicFilters(mat2D<T> * mat, mat2D<T> * F1, mat2D<T> * F2)
	{
		mat2D<T> * temp1 = mat2D<T>::Correlation_Same(mat, mat2D<T>::Rotate180(F1));
		mat2D<T> * result = mat2D<T>::Correlation_Same(temp1, mat2D<T>::Rotate180(F2));
		delete temp1;

		return result;
	}

	static mat2D<T> * Correlation_Same(mat2D<T> * mat, mat2D<T> * kernel)
	{
		int kernelRowTop = (int)ceil(((double)kernel->rows-1) / 2);
		int kernelRowBottom = (int)floor(((double)kernel->rows-1) / 2);
		int kernelColLeft = (int)ceil(((double)kernel->cols-1) / 2);
		int kernelColRight = (int)floor(((double)kernel->cols-1) / 2);
		mat2D<T> * result = new mat2D<T>(mat->rows, mat->cols);
		for (int ir=0; ir < mat->rows; ir++)
		{
			for (int ic=0; ic < mat->cols; ic++)
			{
				T convVal = 0;
				for (int kr=-kernelRowTop; kr <= kernelRowBottom; kr++)
				{
					for (int kc=-kernelColLeft; kc <= kernelColRight; kc++)
					{
						if ((ir+kr >= 0) && (ir+kr < mat->rows) && (ic+kc >= 0) && (ic+kc < mat->cols))
							convVal = convVal + mat->Read(ir+kr, ic+kc) * kernel->Read(kr+kernelRowTop, kc+kernelColLeft);
					}
				}
				result->Write(ir, ic, convVal);
			}
		}
		return result;
	}


	static mat2D<T> * Correlation_Full_basicFilters(mat2D<T> * mat, mat2D<T> * F1, mat2D<T> * F2)
	{
		mat2D<T> * temp1 = mat2D::Correlation_Full(mat, F1);
		mat2D<T> * result = mat2D::Correlation_Full(temp1, F2);
		delete temp1;

		return result;
	}

	static mat2D<T> * Correlation_Full(mat2D<T> * mat, mat2D<T> * kernel)
	{
		mat2D<T> * result = new mat2D<T>(mat->rows+kernel->rows-1, mat->cols+kernel->cols-1);
		for (int ir=0; ir < (mat->rows + kernel->rows - 1); ir++)
		{
			for (int ic=0; ic < (mat->cols + kernel->cols - 1); ic++)
			{
				T convVal = 0;
				for (int kr=0; kr < kernel->rows; kr++)
				{
					for (int kc=0; kc < kernel->cols; kc++)
					{
						if ((ir+kr-kernel->rows+1 >=0) && (ir+kr-kernel->rows+1 < mat->rows) && (ic+kc-kernel->cols+1 >=0) && (ic+kc-kernel->cols+1 < mat->cols))
							convVal = convVal + mat->Read(ir+kr-kernel->rows+1, ic+kc-kernel->cols+1) * kernel->Read(kr, kc);
					}
				}
				result->Write(ir, ic, convVal);
			}
		}
		return result;
	}

	static mat2D<T> * Padding_Mirror(mat2D<T> * mat, int padSizeRows, int padSizeCols)
	{
		mat2D<T> * result = new mat2D<T>(mat->rows + (2*padSizeRows), mat->cols + (2*padSizeCols));
		for (int row=0; row < result->rows; row++)
		{
			int rowOrig;
			if (row < padSizeRows) 
				rowOrig = padSizeRows - row - 1;
			else if (row > mat->rows + padSizeRows - 1) 
					rowOrig = 2*(mat->rows) + padSizeRows - 1 - row;
			else rowOrig = row - padSizeRows;

			for (int col=0; col < result->cols; col++)
			{
				int colOrig;
				if (col < padSizeCols) 
					colOrig = padSizeCols - col - 1;
				else if (col > mat->cols + padSizeCols - 1) 
						colOrig = 2*(mat->cols) + padSizeCols - 1 - col;
				else colOrig = col - padSizeCols;

				result->Write(row, col, mat->Read(rowOrig, colOrig));
			}
		}
		return result;
	}

};

} // end of namespace jpegio

#endif
