/*
 * Copyright (c) 2021, Peter Abeles. All Rights Reserved.
 *
 * This file is part of Efficient Java Matrix Library (EJML).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.ejml.sparse.csc;

import org.ejml.data.DMatrixSparseCSC;
import org.ejml.ops.MatrixIO;
import org.jetbrains.annotations.NotNull;

import java.io.File;
import java.util.Arrays;
import java.util.Random;

public class BenchmarkUtil {
    @NotNull
    private static String getTmpMatricesDir() {
        return System.getProperty("java.io.tmpdir") + "/randomSparseMatrices";
    }

    public static DMatrixSparseCSC createOrLoadRandomCSCMatrix( int numRows, int numCols, int avgEntries, double minVal, double maxVal, int seed ) throws Throwable {
        String tmpMatricesDir = getTmpMatricesDir();
        String matrixName = String.format("/dscc_rows%dcols%davgEntries%dminVal%smaxVal%sseed%d.bin",
                numRows, numCols, avgEntries, doubleToEscapedString(minVal), doubleToEscapedString(maxVal), seed);
        String matrixPath = getTmpMatricesDir() + matrixName;


        DMatrixSparseCSC result;

        File matrixFile = new File(matrixPath);
        if (matrixFile.exists()) {
            result = MatrixIO.loadBin(matrixPath);
        } else {
            result = RandomMatrices_DSCC.generateUniform(numRows, numCols, avgEntries, minVal, maxVal, new Random(seed));
            new File(tmpMatricesDir).mkdir();
            MatrixIO.saveBin(result, matrixPath);
        }

        return result;
    }

    @NotNull
    private static String doubleToEscapedString( double minVal ) {
        return String.valueOf(minVal).replace('.', '_');
    }

    public static void clearCachedMatrices() {
        Arrays.stream(new File(getTmpMatricesDir()).listFiles()).forEach(File::delete);
    }
}
