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

import org.ejml.data.DMatrixD1;
import org.ejml.data.DMatrixRMaj;
import org.ejml.data.DMatrixSparse;
import org.ejml.data.DMatrixSparseCSC;
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
import static org.junit.jupiter.api.Assertions.assertFalse;

@SuppressWarnings({"UnusedMethod"})
public class TestMaskedOperators_DSCC extends BaseTestMatrixMatrixOpsWithSemiRing_DSCC {
    //      test accumulator
    static DSemiRing semiRing = DSemiRings.PLUS_TIMES;
    private static final boolean[] BOOL_VALUES = {true, false};

    private static Stream<Arguments> primitiveVectorSource() {
        double[] inputVector = new double[7];
        Arrays.fill(inputVector, semiRing.add.id);
        inputVector[3] = 0.5;
        inputVector[0] = 0.6;

        double[] prevResult = new double[7];
        prevResult[2] = 99;
        prevResult[0] = 42;

        PrimitiveDMask.Builder maskBuilder = new PrimitiveDMask.Builder(prevResult);

        Stream.Builder<Arguments> builder = Stream.builder();

        for (boolean negated : BOOL_VALUES) {
            for (boolean replaceOutput : BOOL_VALUES) {
                builder.accept(Arguments.of(inputVector, prevResult, maskBuilder.withNegated(negated).build(), replaceOutput));
            }
        }

        return builder.build();
    }

    private static Stream<Arguments> sparseVectorSource() {
        DMatrixSparseCSC otherMatrix = new DMatrixSparseCSC(1, 7);
        otherMatrix.set(0, 3, 0.5);
        otherMatrix.set(0, 4, 0.6);

        DMatrixSparseCSC prevResult = otherMatrix.copy();
        prevResult.set(0, 2, 99);

        Stream.Builder<Arguments> builder = Stream.builder();

        for (boolean structural : BOOL_VALUES) {
            for (boolean negated : BOOL_VALUES) {
                for (boolean replaceOutput : BOOL_VALUES) {
                    Mask mask = DMasks.builder(prevResult, structural).withNegated(negated).build();
                    builder.accept(Arguments.of(otherMatrix, prevResult, mask, replaceOutput));
                }
            }
        }


        return builder.build();
    }

    private static Stream<Arguments> sparseMatrixSource() {
        DMatrixSparseCSC otherMatrix = new DMatrixSparseCSC(7, 7);
        otherMatrix.set(0, 3, 0.5);
        otherMatrix.set(0, 5, 0.6);

        DMatrixSparseCSC prevResult = new DMatrixSparseCSC(7, 7);
        prevResult.set(0, 0, 99);
        prevResult.set(0, 3, 42);

        Stream.Builder<Arguments> builder = Stream.builder();

        for (boolean structural : BOOL_VALUES) {
            for (boolean negated : BOOL_VALUES) {
                for (boolean replaceOutput : BOOL_VALUES) {
                    Mask mask = DMasks.builder(prevResult, structural).withNegated(negated).build();
                    builder.accept(Arguments.of(otherMatrix, prevResult, mask, replaceOutput));
                }
            }
        }

        return builder.build();
    }

    @ParameterizedTest
    @MethodSource("primitiveVectorSource")
    public void mult_v_A(double[] inputVector, double[] prevResult, PrimitiveDMask mask, boolean replaceOutput) {
        double[] found = prevResult.clone();
        double[] foundWithMask = prevResult.clone();

        found = MatrixVectorMultWithSemiRing_DSCC.mult(inputVector, inputMatrix, found, semiRing);
        foundWithMask = MatrixVectorMultWithSemiRing_DSCC.mult(inputVector, inputMatrix, foundWithMask, semiRing, mask, null, replaceOutput);

        assertMaskedResult(prevResult, found, foundWithMask, mask, semiRing.add.id, replaceOutput);
    }

    @ParameterizedTest
    @MethodSource("primitiveVectorSource")
    public void mult_A_v(double[] inputVector, double[] prevResult, PrimitiveDMask mask, boolean replaceOutput) {
        double[] found = prevResult.clone();
        double[] foundWithMask = prevResult.clone();

        found = MatrixVectorMultWithSemiRing_DSCC.mult(inputMatrix, inputVector, found, semiRing);
        foundWithMask = MatrixVectorMultWithSemiRing_DSCC.mult(inputMatrix, inputVector, foundWithMask, semiRing, mask, null, replaceOutput);

        assertMaskedResult(prevResult, found, foundWithMask, mask, semiRing.add.id, replaceOutput);
    }

    @ParameterizedTest
    @MethodSource("primitiveVectorSource")
    public void mult_Trans_A_v(double[] inputVector, double[] prevResult, PrimitiveDMask mask, boolean replaceOutput) {
        double[] found = prevResult.clone();
        double[] foundWithMask = prevResult.clone();

        found = MatrixVectorMultWithSemiRing_DSCC.multTransA(inputMatrix, inputVector, found, semiRing, null, null, replaceOutput);
        foundWithMask = MatrixVectorMultWithSemiRing_DSCC.multTransA(inputMatrix, inputVector, foundWithMask, semiRing, mask, null, replaceOutput);

        assertMaskedResult(prevResult, found, foundWithMask, mask, semiRing.add.id, replaceOutput);
    }

    // matrix, matrix ops

    @ParameterizedTest
    @MethodSource("sparseVectorSource")
    public void mult_A_B(DMatrixSparseCSC sparseVector, DMatrixSparseCSC prevResult, Mask mask, boolean replaceOutput) {
        DMatrixSparseCSC found = CommonOpsWithSemiRing_DSCC.mult(sparseVector, inputMatrix,
                prevResult.copy(), semiRing, null, null, replaceOutput);
        DMatrixSparseCSC foundWithMask = CommonOpsWithSemiRing_DSCC.mult(sparseVector, inputMatrix,
                prevResult.copy(), semiRing, mask, null, replaceOutput);

        assertMaskedResult(prevResult, found, foundWithMask, mask, semiRing.add.id, replaceOutput);
    }


    @ParameterizedTest
    @MethodSource("sparseMatrixSource")
    public void add_A_B(DMatrixSparseCSC otherMatrix, DMatrixSparseCSC prevResult, Mask mask, boolean replaceOutput) {
        DMatrixSparseCSC found = CommonOpsWithSemiRing_DSCC.add(
                1, otherMatrix, 1, inputMatrix, prevResult.copy(), semiRing, null, null, replaceOutput, null, null);
        DMatrixSparseCSC foundWithMask = CommonOpsWithSemiRing_DSCC.add(
                1, otherMatrix, 1, inputMatrix, prevResult.copy(), semiRing, mask, null, replaceOutput, null, null);

        assertMaskedResult(prevResult, found, foundWithMask, mask, semiRing.add.id, replaceOutput);
    }

    @ParameterizedTest
    @MethodSource("sparseMatrixSource")
    public void elementWiseMult(DMatrixSparseCSC otherMatrix, DMatrixSparseCSC prevResult, Mask mask, boolean replaceOutput) {
        DMatrixSparseCSC found = CommonOpsWithSemiRing_DSCC.elementMult(
                otherMatrix, inputMatrix, prevResult.copy(), semiRing, null, null, replaceOutput, null, null);
        DMatrixSparseCSC foundWithMask = CommonOpsWithSemiRing_DSCC.elementMult(
                otherMatrix, inputMatrix, prevResult.copy(), semiRing, mask, null, replaceOutput, null, null);

        assertMaskedResult(prevResult, found, foundWithMask, mask, semiRing.add.id, replaceOutput);
    }

    @ParameterizedTest
    @MethodSource("sparseVectorSource")
    public void apply(DMatrixSparseCSC vector, DMatrixSparseCSC prevResult, Mask mask, boolean replaceOutput) {
        DBinaryOperator second = (x, y) -> y;
        DMatrixSparseCSC result = CommonOps_DSCC.apply(vector, a -> a * 2, prevResult.copy());
        DMatrixSparseCSC resultWithMask = CommonOps_DSCC.apply(vector, a -> a * 2, prevResult.copy(), mask, second, replaceOutput);

        assertMaskedResult(prevResult, result, resultWithMask, mask, 0, replaceOutput);
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
        boolean replaceOutput = false;
        PrimitiveDMask mask = DMasks.builder(prevResult).withNegated(negatedMask).build();

        DMatrixRMaj result = CommonOps_DSCC.reduceRowWise(matrix, 0, Double::sum, prevResult.copy(), null, null, replaceOutput);
        DMatrixRMaj resultWithMask = CommonOps_DSCC.reduceRowWise(matrix, 0, Double::sum, prevResult.copy(), mask, null, replaceOutput);

        assertMaskedResult(prevResult, result, resultWithMask, mask, 0, replaceOutput);
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
        PrimitiveDMask mask = DMasks.builder(prevResult).withNegated(negatedMask).build();

        boolean replaceOutput = false;
        DMatrixRMaj result = CommonOps_DSCC.reduceColumnWise(matrix, 0, Double::sum, prevResult.copy(), null, null, replaceOutput);
        DMatrixRMaj resultWithMask = CommonOps_DSCC.reduceColumnWise(matrix, 0, Double::sum, prevResult.copy(), mask, null, replaceOutput);

        assertMaskedResult(prevResult, result, resultWithMask, mask, 0, replaceOutput);
    }


    private void assertMaskedResult(DMatrixSparseCSC prevResult, DMatrixSparseCSC found, DMatrixSparseCSC foundWithMask,
                                    Mask mask, double id, boolean replaceOutput) {
        Iterator<DMatrixSparse.CoordinateRealValue> it = found.createCoordinateIterator();
        // check that existing result were not overwritten
        it.forEachRemaining(value -> {
            if (mask.isSet(value.row, value.col)) {
                assertEquals(found.get(value.row, value.col), foundWithMask.get(value.row, value.col), "Field should have been computed");
            } else if (!replaceOutput) {
                assertEquals(prevResult.get(value.row, value.col), foundWithMask.get(value.row, value.col), "Field from initial result was overwritten");
            } else if (!mask.isSet(value.row, value.col)) {
                assertEquals(id, foundWithMask.get(value.row, value.col), "Field shouldn't be computed");
            }
        });

        // checking that untouched cells are still present
        prevResult.createCoordinateIterator().forEachRemaining(value -> {
            if (!mask.isSet(value.row, value.col) && !replaceOutput) {
                assertEquals(prevResult.get(value.row, value.col), foundWithMask.get(value.row, value.col), "Field from initial result was deleted");
            } else if (replaceOutput && !found.isAssigned(value.row, value.col)) {
                assertFalse(foundWithMask.isAssigned(value.row, value.col), "Field from initial result was not cleared");
            }
        });
    }

    private void assertMaskedResult(DMatrixD1 prevResult, DMatrixD1 found, DMatrixD1 foundWithMask,
                                    Mask mask, double id, boolean replaceOutput) {
        for (int row = 0; row < found.getNumRows(); row++) {
            for (int col = 0; col < found.getNumCols(); col++) {
                if (mask.isSet(row, col) && replaceOutput) {
                    assertEquals(found.get(row, col), foundWithMask.get(row, col), "Field should have been computed/Initial result cleared");
                } else if (replaceOutput && !mask.isSet(row, col)) {
                    assertEquals(id, foundWithMask.get(row, col), "Field should not be computed");
                } else if (!mask.isSet(row, col)) {
                    assertEquals(prevResult.get(row, col), foundWithMask.get(row, col), "Field from initial result was overwritten");
                }
            }
        }
    }

    private void assertMaskedResult(double[] prevResult, double[] found, double[] foundWithMask,
                                    PrimitiveDMask mask, double id, boolean replaceOutput) {
        for (int i = 0; i < found.length; i++) {
            if (mask.isSet(i)) {
                assertEquals(foundWithMask[i], found[i], "Computation differs");
            } else if (!mask.isSet(i) && replaceOutput) {
                assertEquals(id, foundWithMask[i], "Wrongly computed");
            } else {
                assertEquals(prevResult[i], foundWithMask[i], "Initial result was overwritten");
            }
        }
    }
}
