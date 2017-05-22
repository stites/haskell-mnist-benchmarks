{-# LANGUAGE OverloadedLists #-}
module MNIST.Tensorflow where

import MNIST.Prelude
import MNIST.DataSet

import qualified Data.Text.IO as T
import qualified Data.Vector  as V
import qualified TensorFlow.Gradient as TF
import qualified TensorFlow.Core     as TF
import qualified TensorFlow.Ops      as TF hiding (initializedVariable, zeroInitializedVariable)
import qualified TensorFlow.Variable as TF

numPixels = fromIntegral nPixels :: Int64
numLabels = fromIntegral nLabels :: Int64

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
getBatchSize Variable      = -1

-- createModel :: BatchSize -> Int64 -> TF.Build Model
-- createModel (getBatchSize -> batchSize) hiddenSize = do
--   images            <- TF.placeholder [batchSize, numPixels]
--   (hWeights, hBias) <- initWeightsAndBias numPixels hiddenSize
--   let hidden = TF.relu $ (images <* hWeights) <+ hBias
--
--   -- Logits
--   (logitWeights, logitBias) <- initWeightsAndBias hiddenSize numLabels
--   let logits = (hidden <* logitWeights) <+ logitBias
--
--   predict <- TF.render . TF.cast $
--                 TF.argMax (TF.softmax logits) (TF.scalar (1 :: LabelType))
--
--   -- Create training action.
--   labels <- TF.placeholder [batchSize]
--
--   let labelVecs = TF.oneHot labels (fromIntegral numLabels) 1 0
--       -- loss :: Tensor Build Float
--       loss = reduceMean $ fst $ TF.softmaxCrossEntropyWithLogits logits labelVecs
--       -- params :: [TF.Variable Float]
--       params = [hWeights, hBias, logitWeights, logitBias]
--
--   grads <- TF.gradients loss params
--
--   let lr = TF.scalar 0.00001
--       applyGrad param grad = TF.assignAdd param (negate $ lr `TF.mul` grad)
--
--   trainStep <- TF.group =<< zipWithM applyGrad params grads
--
--   let correctPredictions = TF.equal predict labels
--   errorRateTensor <- TF.render $ 1 - reduceMean (TF.cast correctPredictions)
--
--   return Model {
--         train = \imFeed lFeed -> TF.runWithFeeds_ [
--               TF.feed images imFeed
--             , TF.feed labels lFeed
--             ] trainStep
--       , infer = \imFeed -> TF.runWithFeeds [TF.feed images imFeed] predict
--       , errorRate = \imFeed lFeed -> TF.unScalar <$> TF.runWithFeeds [
--               TF.feed images imFeed
--             , TF.feed labels lFeed
--             ] errorRateTensor
--       }
--   where
--     -- initWeightsAndBias :: Int64 -> Int64 -> TF.Build (Tensor z Float, Tensor z Float)
--     initWeightsAndBias x y = do
--       vec  <- TF.initializedVariable =<< randomParam x [x, y]
--       bias <- TF.zeroInitializedVariable [y]
--       return (vec, bias)
--
--     -- (<*) :: Tensor v Float -> TF.Variable Float -> Tensor Build Float
--     (<*) a b = a `TF.matMul` TF.readValue b
--
--     -- (<+) :: Tensor Build Float -> TF.Variable Float -> Tensor Build Float
--     (<+) a b = a `TF.add` TF.readValue b
--
createModel :: TF.Build Model
createModel = do
    -- Use -1 batch size to support variable sized batches.
    let batchSize = -1
    -- Inputs.
    images <- TF.placeholder [batchSize, numPixels]
    -- Hidden layer.
    let numUnits = 500
    (hiddenWeights :: TF.Variable Float) <-
        TF.initializedVariable =<< randomParam numPixels [numPixels, numUnits]
    (hiddenBiases :: TF.Variable Float) <- TF.zeroInitializedVariable [numUnits]
    let hiddenZ = (images `TF.matMul` TF.readValue hiddenWeights)
                  `TF.add` TF.readValue hiddenBiases
    let hidden = TF.relu hiddenZ
    -- Logits.
    (logitWeights :: TF.Variable Float) <-
        TF.initializedVariable =<< randomParam numUnits [numUnits, numLabels]
    (logitBiases :: TF.Variable Float) <- TF.zeroInitializedVariable [numLabels]
    let logits = (hidden `TF.matMul` TF.readValue logitWeights)
                 `TF.add` TF.readValue logitBiases
    predict <- TF.render $ TF.cast $
               TF.argMax (TF.softmax logits) (TF.scalar (1 :: LabelType))

    -- Create training action.
    labels <- TF.placeholder [batchSize]
    let labelVecs = TF.oneHot labels (fromIntegral numLabels) 1 0
        loss   = reduceMean . fst $ TF.softmaxCrossEntropyWithLogits logits labelVecs
        params :: [TF.Variable Float]
        params = [hiddenWeights, hiddenBiases, logitWeights, logitBiases]

    grads <- undefined -- TF.gradients loss params

    let lr = TF.scalar 0.00001
        applyGrad param grad = TF.assignAdd param (negate $ lr `TF.mul` grad)
    trainStep <- TF.group =<< zipWithM applyGrad params grads

    let correctPredictions = TF.equal predict labels
    errorRateTensor <- TF.render $ 1 - reduceMean (TF.cast correctPredictions)

    return Model {
          train = \imFeed lFeed -> TF.runWithFeeds_ [
                TF.feed images imFeed
              , TF.feed labels lFeed
              ] trainStep
        , infer = \imFeed -> TF.runWithFeeds [TF.feed images imFeed] predict
        , errorRate = \imFeed lFeed -> TF.unScalar <$> TF.runWithFeeds [
                TF.feed images imFeed
              , TF.feed labels lFeed
              ] errorRateTensor
        }

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
    model <- TF.build createModel

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

