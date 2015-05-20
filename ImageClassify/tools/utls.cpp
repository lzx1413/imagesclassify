
#include "utls.h"
#include <QDebug>
void PrintRandAay(int i_Array[], int count)
{
	for (int i = 0; i < count; i++)
	{
		qDebug() << i_Array[i];
	}
	return;
}


void GetRandArray(int i_Array[], int count)
{
	int Number = 0;
	for (int i = 0;; i++)
	{
		int i_Value = rand() % 20;                     
		if (CheckArray(i_Value, i_Array, Number))   
		{
			i_Array[Number] = i_Value;            
			Number++;
			if (Number == count)                   
				return;
		}

	}
	return;
}


int CheckArray(int i_Value, int i_Array[], int count)
{
	int result = 1;

	for (int i = 0; i < count; i++)
	{
		if (i_Value == i_Array[i])
			return 0;
	}
	return 1;
}
