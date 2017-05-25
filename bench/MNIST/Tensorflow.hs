{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE NamedFieldPuns #-}
module Main where

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


data MNISTVariables = MNISTVariables
  { images :: Tensor TF.Value Float
  , hiddenLayer :: (Tensor TF.Ref Float, Tensor TF.Ref Float)
  , hiddenAction :: Tensor Build Float
  , logitLayer  :: (Tensor TF.Ref Float, Tensor TF.Ref Float)
  , logitAction :: Tensor Build Float
  , labels :: Tensor TF.Value LabelType
  , predict :: Tensor TF.Value LabelType
  }


mkVariables :: BatchSize -> Int64 -> TF.Build MNISTVariables
mkVariables (getBatchSize->batchSize) numHidden = do
    -- Inputs.
    images <- TF.placeholder [batchSize, numPixels]

    h@(hWeights, hBiases) <- initialize numPixels numHidden
    let hidden = TF.relu $ (images `TF.matMul` hWeights) `TF.add` hBiases

    l@(logitWeights, logitBiases) <- initialize numHidden numLabels
    let logit = (hidden `TF.matMul` logitWeights) `TF.add` logitBiases

    labels <- TF.placeholder [batchSize]

    predict <- TF.render . TF.cast $
      TF.argMax (TF.softmax logit) (TF.scalar (1 :: LabelType))

    return MNISTVariables
      { images = images
      , hiddenLayer = h
      , hiddenAction = hidden
      , logitLayer = l
      , logitAction = logit
      , labels = labels
      , predict = predict
      }

    where
      initialize :: Int64 -> Int64 -> Build (Tensor TF.Ref Float, Tensor TF.Ref Float)
      initialize x y = do
        weights <- TF.initializedVariable =<< randomParam x [x, y]
        bias <- TF.zeroInitializedVariable [y]
        return (weights, bias)


infer' :: ReaderT MNISTVariables Build (TensorData Float -> Session (Vector LabelType))
infer' = do
  MNISTVariables{images, predict} <- ask
  return $ \imFeed -> TF.runWithFeeds [TF.feed images imFeed] predict


training' :: ReaderT MNISTVariables Build (TensorData Float -> TensorData LabelType -> Session ())
training' = do
  MNISTVariables{images, labels, hiddenLayer, logitLayer, logitAction} <- ask
  let
    (hWeights, hBiases) = hiddenLayer
    (logitWeights, logitBiases) = logitLayer

    labelVecs :: Tensor Build Float
    labelVecs = TF.oneHot labels (fromIntegral numLabels) 1 0

    loss :: Tensor Build Float
    loss = reduceMean . fst $ TF.softmaxCrossEntropyWithLogits logitAction labelVecs

    params :: [Tensor TF.Ref Float]
    params = [hWeights, hBiases, logitWeights, logitBiases]

  grads     <- lift $ TF.gradients loss params
  trainStep <- lift $ applyGradients params grads

  return $ \imFeed lFeed ->
    TF.runWithFeeds_
      [ TF.feed images imFeed
      , TF.feed labels lFeed
      ] trainStep

  where
    applyGradients :: [Tensor TF.Ref Float] -> [Tensor TF.Value Float] -> Build TF.ControlNode
    applyGradients params grads = zipWithM applyGrad params grads >>= TF.group

    applyGrad :: Tensor TF.Ref Float -> Tensor TF.Value Float -> Build (Tensor TF.Ref Float)
    applyGrad param grad = TF.assign param $ param `TF.sub` (lr `TF.mul` grad)

    lr :: Tensor Build Float
    lr = TF.scalar 0.00001


errorRate' :: ReaderT MNISTVariables Build (TensorData Float -> TensorData LabelType -> Session Float)
errorRate' = do
  MNISTVariables{images, predict, labels} <- ask
  let correctPredictions = TF.equal predict labels
  errorRateTensor <- lift . TF.render $ 1 - reduceMean (TF.cast correctPredictions)

  return $ \imFeed lFeed ->
    TF.unScalar <$> TF.runWithFeeds
      [ TF.feed images imFeed
      , TF.feed labels lFeed
      ] errorRateTensor


createModel :: BatchSize -> Int64 -> TF.Build Model
createModel batchSize numHidden = do
    -- Inputs.
    vars      <- mkVariables batchSize numHidden
    inference <- runReaderT infer'     vars
    training  <- runReaderT training'  vars
    errorRate <- runReaderT errorRate' vars

    return Model
      { train     = training
      , infer     = inference
      , errorRate = errorRate
      }


main :: IO ()
main = TF.runSession $ do
    -- Read training and test data.
    !trainingSet <- liftIO trainingDataTf
    !testSet     <- liftIO testDataTf

    let
      trainingImages = fmap snd trainingSet
      trainingLabels = fmap fst trainingSet
      testImages = fmap snd testSet
      testLabels = fmap fst testSet

    -- Create the model.
    model <- TF.build $ createModel Variable 500

    -- Train.
    forM_ ([0..1000] :: [Int]) $ \i -> do
        let images = encodeImageBatch (selectBatch i trainingImages)
            labels = encodeLabelBatch (selectBatch i trainingLabels)
        train model images labels
        when (i `mod` 100 == 0) $ do
            err <- errorRate model images labels
            liftIO $ putStrLn $ "training error " ++ show (err * 100)

    -- Test.
    testErr <- errorRate model (encodeImageBatch testImages)
                               (encodeLabelBatch testLabels)
    liftIO $ putStrLn $ "\n" ++ "test error " ++ show (testErr * 100)

    -- Show some predictions.
    testPreds <- infer model (encodeImageBatch testImages)

    liftIO $ forM_ ([0..3] :: [Int]) $ \i -> do
        -- T.putStrLn $ drawMNIST $ testImages !! i
        putStrLn $ "\n" ++ "expected " ++ show (testLabels !! i)
        putStrLn $         "     got " ++ show (testPreds V.! i)

  where
    batchSize :: Int
    batchSize = 100

    -- Functions for generating batches.
    encodeImageBatch :: [Vector Int] -> TensorData Float
    encodeImageBatch xs =
        TF.encodeTensorData [genericLength xs, numPixels]
                            (fromIntegral <$> mconcat xs)

    encodeLabelBatch :: [Int] -> TensorData LabelType
    encodeLabelBatch xs =
      TF.encodeTensorData [genericLength xs]
                          (fromIntegral <$> V.fromList xs)

    selectBatch :: Int -> [a] -> [a]
    selectBatch i xs = take batchSize $ drop (i * batchSize) (cycle xs)
