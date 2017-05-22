{-# LANGUAGE OverloadedLists #-}
module MNIST.Tensorflow where

import MNIST.Prelude
import MNIST.DataSet

import qualified Data.Vector  as V
import qualified TensorFlow.Gradient  as TF
import qualified TensorFlow.Core      as TF
import qualified TensorFlow.Ops       as TF


numPixels :: Int64
numPixels = fromIntegral nPixels
numLabels :: Int64
numLabels = fromIntegral nLabels

-- | Create tensor with random values where the stddev depends on the width.
randomParam :: Int64 -> Shape -> Build (Tensor Build Float)
randomParam width (TF.Shape shape) =
  TF.truncatedNormal (TF.vector shape) >>= pure . (`TF.mul` stddev)
  where
    stddev :: Tensor Build Float
    stddev = TF.scalar (1 / sqrt (fromIntegral width))


reduceMean :: Tensor Build Float -> Tensor Build Float
reduceMean xs = TF.mean xs (TF.scalar (0 :: Int32))


-- Types must match due to model structure.
type LabelType = Int32

data Model = Model
  { train :: TensorData Float -> TensorData LabelType -> Session ()
          -- ^ mnist images and labels
  , infer :: TensorData Float -> Session (V.Vector LabelType)
          -- ^ given mnist images, output a predictions
  , errorRate :: TensorData Float -> TensorData LabelType -> Session Float
          -- ^ show us the error rate
  }


data BatchSize = BatchSize Int64 | Variable


getBatchSize :: BatchSize -> Int64
getBatchSize (BatchSize i) =  i
getBatchSize Variable      = -1   -- ^ Use -1 batch size to support variable sized batches.


createModel :: BatchSize -> Int64 -> TF.Build Model
createModel (getBatchSize->batchSize) numHidden = do
    -- Inputs.
    images :: Tensor TF.Value Float <- TF.placeholder [batchSize, numPixels]

    -- Hidden layer.
    (hWeights, hBiases) <- initialize numPixels numHidden
    let hidden :: Tensor Build Float
        hidden = TF.relu $ (images `TF.matMul` hWeights) `TF.add` hBiases

    -- Logits.
    (logitWeights, logitBiases) <- initialize numHidden numLabels
    let logits :: Tensor Build Float
        logits = (hidden `TF.matMul` logitWeights) `TF.add` logitBiases

    predict <- calculatePrediction logits

    let
      inferAction :: TensorData Float -> Session (Vector LabelType)
      inferAction imFeed = TF.runWithFeeds [TF.feed images imFeed] predict

    -- ========================================================================= --

    -- Create training action.
    labels <- TF.placeholder [batchSize]

    let
      labelVecs :: Tensor Build Float
      labelVecs = TF.oneHot labels (fromIntegral numLabels) 1 0

      loss :: Tensor Build Float
      loss = reduceMean $ fst $ TF.softmaxCrossEntropyWithLogits logits labelVecs

      params :: [Tensor TF.Ref Float]
      params = [hWeights, hBiases, logitWeights, logitBiases]

    grads     <- TF.gradients loss params
    trainStep <- applyGradients params grads

    let
      trainingAction :: TensorData Float -> TensorData LabelType -> Session ()
      trainingAction imFeed lFeed = TF.runWithFeeds_
        [ TF.feed images imFeed
        , TF.feed labels lFeed
        ] trainStep

    -- ========================================================================= --

    let correctPredictions = TF.equal predict labels

    errorRateTensor <- TF.render $ 1 - reduceMean (TF.cast correctPredictions)

    let
      errorRateAction :: TensorData Float -> TensorData LabelType -> Session Float
      errorRateAction imFeed lFeed = TF.unScalar <$> TF.runWithFeeds
        [ TF.feed images imFeed
        , TF.feed labels lFeed
        ] errorRateTensor

    return Model
      { train = trainingAction
      , infer = inferAction
      , errorRate = errorRateAction
      }

    where
      initialize :: Int64 -> Int64 -> Build (Tensor TF.Ref Float, Tensor TF.Ref Float)
      initialize x y = do
        weights <- TF.initializedVariable =<< randomParam x [x, y]
        bias <- TF.zeroInitializedVariable [y]
        return (weights, bias)

      calculatePrediction :: Tensor Build Float -> Build (Tensor TF.Value LabelType)
      calculatePrediction logits = TF.render . TF.cast $
        TF.argMax (TF.softmax logits) (TF.scalar (1 :: LabelType))

      applyGradients :: [Tensor TF.Ref Float] -> [Tensor TF.Value Float] -> Build TF.ControlNode
      applyGradients params grads = zipWithM applyGrad params grads >>= TF.group
        where
          applyGrad :: Tensor TF.Ref Float -> Tensor TF.Value Float -> Build (Tensor TF.Ref Float)
          applyGrad param grad = TF.assign param $ param `TF.sub` (lr `TF.mul` grad)

          lr :: Tensor Build Float
          lr = TF.scalar 0.00001


main :: IO ()
main = TF.runSession $ do
    -- Read training and test data.
    trainingSet <- liftIO trainingData
    testSet     <- liftIO testData
    let
      trainingImages = fmap snd trainingSet
      trainingLabels = fmap fst trainingSet
      testImages = fmap snd testSet
      testLabels = fmap fst testSet


    -- Create the model.
    model <- TF.build $ createModel Variable 500

    -- Functions for generating batches.
    let
      encodeImageBatch :: [Vector Int] -> TensorData Float
      encodeImageBatch xs =
            TF.encodeTensorData [genericLength xs, numPixels]
                                (fromIntegral <$> mconcat xs)
    let encodeLabelBatch xs =
            TF.encodeTensorData [genericLength xs]
                                (fromIntegral <$> V.fromList xs)
    let batchSize = 100
    let selectBatch :: Int -> [a] -> [a]
        selectBatch i xs = take batchSize $ drop (i * batchSize) (cycle xs)

    -- Train.
    forM_ ([0..1000] :: [Int]) $ \i -> do
        let images = encodeImageBatch (selectBatch i trainingImages)
            labels = encodeLabelBatch (selectBatch i trainingLabels)
        train model images labels
        when (i `mod` 100 == 0) $ do
            err <- errorRate model images labels
            liftIO $ putStrLn $ "training error " ++ show (err * 100)
    liftIO $ putStrLn ""

    -- Test.
    testErr <- errorRate model (encodeImageBatch testImages)
                               (encodeLabelBatch testLabels)
    liftIO $ putStrLn $ "test error " ++ show (testErr * 100)

    -- Show some predictions.
    testPreds <- infer model (encodeImageBatch testImages)
    liftIO $ forM_ ([0..3] :: [Int]) $ \i -> do
        putStrLn ""
        -- T.putStrLn $ drawMNIST $ testImages !! i
        putStrLn $ "expected " ++ show (testLabels !! i)
        putStrLn $ "     got " ++ show (testPreds V.! i)

