/*
 * Copyright (c) 2009-2020, Peter Abeles. All Rights Reserved.
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

package org.ejml.sparse.csc.mult;

import org.ejml.data.*;
import org.ejml.masks.DMasks;
import org.ejml.masks.Mask;
import org.ejml.masks.PrimitiveDMask;
import org.ejml.ops.DBinaryOperator;
import org.ejml.ops.DSemiRing;
import org.ejml.ops.DSemiRings;
import org.ejml.sparse.csc.CommonOpsWithSemiRing_DSCC;
import org.ejml.sparse.csc.CommonOps_DSCC;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Arrays;
import java.util.Iterator;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class TestMaskedOperators_DSCC extends BaseTestMatrixMatrixOpsWithSemiRing_DSCC {

    static DSemiRing semiRing = DSemiRings.PLUS_TIMES;

    private static Stream<Arguments> primitiveVectorSource() {
        double[] inputVector = new double[7];
        Arrays.fill(inputVector, semiRing.add.id);
        inputVector[3] = 0.5;
        inputVector[0] = 0.6;

        double[] prevResult = new double[7];
        prevResult[3] = 99;
        prevResult[0] = 42;

        return Stream.of(
                Arguments.of(inputVector, prevResult, new PrimitiveDMask(prevResult, true)),
                Arguments.of(inputVector, prevResult, new PrimitiveDMask(prevResult, false))
        );
    }

    private static Stream<Arguments> sparseVectorSource() {
        DMatrixSparseCSC otherMatrix = new DMatrixSparseCSC(1, 7);
        otherMatrix.set(0, 3, 0.5);
        otherMatrix.set(0, 4, 0.6);

        DMatrixSparseCSC prevResult = otherMatrix.copy();
        prevResult.set(0, 2, 99);

        return Stream.of(
                Arguments.of(otherMatrix, prevResult, DMasks.of(prevResult, false, true)),
                Arguments.of(otherMatrix, prevResult, DMasks.of(prevResult, true, false)),
                Arguments.of(otherMatrix, prevResult, DMasks.of(prevResult, false, false)),
                Arguments.of(otherMatrix, prevResult, DMasks.of(prevResult, true, true))
        );
    }

    private static Stream<Arguments> sparseMatrixSource() {
        DMatrixSparseCSC otherMatrix = new DMatrixSparseCSC(7, 7);
        otherMatrix.set(0, 3, 0.5);
        otherMatrix.set(0, 5, 0.6);

        DMatrixSparseCSC prevResult = new DMatrixSparseCSC(7, 7);
        prevResult.set(0, 0, 99);
        prevResult.set(0, 3, 42);

        return Stream.of(
                Arguments.of(otherMatrix, prevResult, DMasks.of(prevResult, false, true)),
                Arguments.of(otherMatrix, prevResult, DMasks.of(prevResult, true, false)),
                Arguments.of(otherMatrix, prevResult, DMasks.of(prevResult, false, false)),
                Arguments.of(otherMatrix, prevResult, DMasks.of(prevResult, true, true))
        );
    }

    @ParameterizedTest
    @MethodSource("primitiveVectorSource")
    public void mult_v_A(double[] inputVector, double[] prevResult, PrimitiveDMask mask) {
        double[] found = prevResult.clone();
        double[] foundWithMask = prevResult.clone();

        MatrixVectorMultWithSemiRing_DSCC.mult(inputVector, inputMatrix, found, semiRing);
        MatrixVectorMultWithSemiRing_DSCC.mult(inputVector, inputMatrix, foundWithMask, semiRing, mask);

        assertMaskedResult(prevResult, found, foundWithMask, mask);
    }

    @ParameterizedTest
    @MethodSource("primitiveVectorSource")
    public void mult_A_v(double[] inputVector, double[] prevResult, PrimitiveDMask mask) {
        double[] found = prevResult.clone();
        double[] foundWithMask = prevResult.clone();

        MatrixVectorMultWithSemiRing_DSCC.mult(inputMatrix, inputVector, found, semiRing);
        MatrixVectorMultWithSemiRing_DSCC.mult(inputMatrix, inputVector, foundWithMask, semiRing, mask);

        assertMaskedResult(prevResult, found, foundWithMask, mask);
    }

    // matrix, matrix ops

    @ParameterizedTest
    @MethodSource("sparseVectorSource")
    public void mult_A_B(DMatrixSparseCSC sparseVector, DMatrixSparseCSC prevResult, Mask mask) {
        DMatrixSparseCSC found = CommonOpsWithSemiRing_DSCC.mult(sparseVector, inputMatrix, prevResult.copy(), semiRing, null, null);
        DMatrixSparseCSC foundWithMask = CommonOpsWithSemiRing_DSCC.mult(sparseVector, inputMatrix, prevResult.copy(), semiRing, mask, null);

        assertMaskedResult(prevResult, found, foundWithMask, mask);
    }

    // Todo: solve problem with non-negated masks

    @ParameterizedTest
    @MethodSource("sparseVectorSource")
    public void mult_T_A_B(DMatrixSparseCSC sparseVector, DMatrixSparseCSC prevResult, Mask mask) {
        DMatrixSparseCSC transposed_vector = CommonOps_DSCC.transpose(sparseVector, null, null);

        DMatrixSparseCSC foundTA = CommonOpsWithSemiRing_DSCC.multTransA(
                transposed_vector, inputMatrix, prevResult.copy(), semiRing, null, null, null, null);
        DMatrixSparseCSC foundTAWithMask = CommonOpsWithSemiRing_DSCC.multTransA(
                transposed_vector, inputMatrix, prevResult.copy(), semiRing, mask, null, null, null);

        assertMaskedResult(prevResult, foundTA, foundTAWithMask, mask);
    }

    @ParameterizedTest
    @MethodSource("sparseVectorSource")
    public void mult_A_T_B(DMatrixSparseCSC sparseVector, DMatrixSparseCSC prevResult, Mask mask) {
        DMatrixSparseCSC transposed_matrix = CommonOps_DSCC.transpose(inputMatrix, null, null);

        DMatrixSparseCSC foundTB = CommonOpsWithSemiRing_DSCC.multTransB(
                sparseVector, transposed_matrix, prevResult.copy(), semiRing, null, null, null, null);
        DMatrixSparseCSC foundTBWithMask = CommonOpsWithSemiRing_DSCC.multTransB(
                sparseVector, transposed_matrix, prevResult.copy(), semiRing, mask, null, null, null);

        assertMaskedResult(prevResult, foundTB, foundTBWithMask, mask);
    }


    @ParameterizedTest
    @MethodSource("sparseMatrixSource")
    public void add_A_B(DMatrixSparseCSC otherMatrix, DMatrixSparseCSC prevResult, Mask mask) {
        DMatrixSparseCSC found = CommonOpsWithSemiRing_DSCC.add(
                1, otherMatrix, 1, inputMatrix, prevResult.copy(), semiRing, null, null, null, null);
        DMatrixSparseCSC foundWithMask = CommonOpsWithSemiRing_DSCC.add(
                1, otherMatrix, 1, inputMatrix, prevResult.copy(), semiRing, mask, null, null, null);

        assertMaskedResult(prevResult, found, foundWithMask, mask);
    }

    @ParameterizedTest
    @MethodSource("sparseMatrixSource")
    public void elementWiseMult(DMatrixSparseCSC otherMatrix, DMatrixSparseCSC prevResult, Mask mask) {
        DMatrixSparseCSC found = CommonOpsWithSemiRing_DSCC.elementMult(
                otherMatrix, inputMatrix, prevResult.copy(), semiRing, null, null, null, null);
        DMatrixSparseCSC foundWithMask = CommonOpsWithSemiRing_DSCC.elementMult(
                otherMatrix, inputMatrix, prevResult.copy(), semiRing, mask, null, null, null);

        assertMaskedResult(prevResult, found, foundWithMask, mask);
    }

    @ParameterizedTest
    @MethodSource("sparseVectorSource")
    public void apply(DMatrixSparseCSC vector, DMatrixSparseCSC prevResult, Mask mask) {
        DBinaryOperator second = (x, y) -> y;
        DMatrixSparseCSC result = CommonOps_DSCC.apply(vector, a -> a * 2, prevResult.copy(), null, null);
        DMatrixSparseCSC resultWithMask = CommonOps_DSCC.apply(vector, a -> a * 2, prevResult.copy(), mask, second);

        assertMaskedResult(prevResult, result, resultWithMask, mask);
    }

    @ParameterizedTest
    @ValueSource(strings = {"true", "false"})
    public void reduceRowWise(boolean negatedMask) {
        DMatrixSparseCSC matrix = new DMatrixSparseCSC(7, 7);
        matrix.set(0, 3, 0.5);
        matrix.set(0, 4, 0.6);
        matrix.set(4, 0, 0.1);
        matrix.set(3, 0, 0.2);

        double[] prevPrimitiveResult = new double[7];
        prevPrimitiveResult[2] = 42;
        prevPrimitiveResult[0] = 99;

        DMatrixRMaj prevResult = DMatrixRMaj.wrap(matrix.numRows, 1, prevPrimitiveResult);
        Mask mask = DMasks.of(prevResult, negatedMask);

        DMatrixRMaj result = CommonOps_DSCC.reduceRowWise(matrix, 0, Double::sum, prevResult.copy(), null, null);
        DMatrixRMaj resultWithMask = CommonOps_DSCC.reduceRowWise(matrix, 0, Double::sum, prevResult.copy(), mask, null);

        assertMaskedResult(prevResult, result, resultWithMask, mask);
    }

    @ParameterizedTest
    @ValueSource(strings = {"true", "false"})
    public void reduceColumnWise(boolean negatedMask) {
        DMatrixSparseCSC matrix = new DMatrixSparseCSC(7, 7);
        matrix.set(0, 3, 0.5);
        matrix.set(0, 4, 0.6);
        matrix.set(4, 0, 0.1);
        matrix.set(3, 0, 0.2);

        double[] prevPrimitiveResult = new double[7];
        prevPrimitiveResult[2] = 42;
        prevPrimitiveResult[0] = 99;

        DMatrixRMaj prevResult = DMatrixRMaj.wrap(1, matrix.numCols, prevPrimitiveResult);
        Mask mask = DMasks.of(prevResult, negatedMask);

        DMatrixRMaj result = CommonOps_DSCC.reduceColumnWise(matrix, 0, Double::sum, prevResult.copy(), null, null);
        DMatrixRMaj resultWithMask = CommonOps_DSCC.reduceColumnWise(matrix, 0, Double::sum, prevResult.copy(), mask, null);

        assertMaskedResult(prevResult, result, resultWithMask, mask);
    }


    private void assertMaskedResult(DMatrixSparseCSC prevResult, DMatrixSparseCSC found, DMatrixSparseCSC foundWithMask, Mask mask) {
        Iterator<DMatrixSparse.CoordinateRealValue> it = found.createCoordinateIterator();
        // check that existing result were not overwritten
        it.forEachRemaining(value -> {
            if (mask.isSet(value.row, value.col)) {
                assertEquals(found.get(value.row, value.col), foundWithMask.get(value.row, value.col), "Field should have been computed");
            } else {
                assertEquals(prevResult.get(value.row, value.col), foundWithMask.get(value.row, value.col), "Field from initial result was overwritten");
            }
        });

        // checking that untouched cells are still present
        prevResult.createCoordinateIterator().forEachRemaining(value -> {
            if (!mask.isSet(value.row, value.col)) {
                assertEquals(prevResult.get(value.row, value.col), foundWithMask.get(value.row, value.col), "Field from initial result was deleted");
            }
        });
    }

    private void assertMaskedResult(DMatrixD1 prevResult, DMatrixD1 found, DMatrixD1 foundWithMask, Mask mask) {
        for (int row = 0; row < found.getNumRows(); row++) {
            for (int col = 0; col < found.getNumCols(); col++) {
                if (mask.isSet(row, col)) {
                    assertEquals(found.get(row, col), foundWithMask.get(row, col), "Field should have been computed");
                } else {
                    assertEquals(prevResult.get(row, col), foundWithMask.get(row, col), "Field from initial result was overwritten");
                }
            }
        }
    }

    private void assertMaskedResult(double[] prevResult, double[] found, double[] foundWithMask, PrimitiveDMask mask) {
        for (int i = 0; i < found.length; i++) {
            if (mask.isSet(i)) {
                assertEquals(foundWithMask[i], found[i], "Computation differs");
            } else {
                assertEquals(prevResult[i], foundWithMask[i], "Initial result was overwritten");
            }
        }
    }
}