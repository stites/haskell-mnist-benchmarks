{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ConstraintKinds #-}
{-# OPTIONS_GHC -fno-warn-orphans #-}
module MNIST.Backprop where

import MNIST.Prelude
import qualified Data.Vector.Generic    as VG
import qualified Data.Vector.Unboxed    as VU
import qualified Generics.SOP           as SOP
import qualified Numeric.LinearAlgebra  as HM
import qualified System.Random.MWC      as MWC
import Numeric.LinearAlgebra.Static hiding (dot)
import Numeric.Backprop

-- | type-combinators alias for the terminal constructor
nil :: Prod f '[]
nil = Ø

-- ========================================================================= --
-- Internal structure of our FF neural net

data Layer i o = Layer
  { _lWeights :: !(L o i)
  , _lBiases  :: !(R o)
  } deriving (Show, Generic)

instance SOP.Generic (Layer i o)
instance NFData (Layer i o)

data Network i h1 h2 o = Net
  { _nLayer1 :: !(Layer i  h1)
  , _nLayer2 :: !(Layer h1 h2)
  , _nLayer3 :: !(Layer h2 o)
  } deriving (Show, Generic)

instance SOP.Generic (Network i h1 h2 o)
instance NFData (Network i h1 h2 o)

-- ========================================================================= --
-- type classes

type KnownNat4 i h1 h2 o = (KnownNat i, KnownNat h1, KnownNat h2, KnownNat o)
type KnownNat2 i o = (KnownNat i, KnownNat o)


instance KnownNat2 i o => Num (Layer i o) where
  Layer w1 b1 + Layer w2 b2 = Layer (w1 + w2) (b1 + b2)
  Layer w1 b1 - Layer w2 b2 = Layer (w1 - w2) (b1 - b2)
  Layer w1 b1 * Layer w2 b2 = Layer (w1 * w2) (b1 * b2)
  abs    (Layer w b)        = Layer (abs    w) (abs    b)
  signum (Layer w b)        = Layer (signum w) (signum b)
  negate (Layer w b)        = Layer (negate w) (negate b)
  fromInteger x = Layer (fromInteger x) (fromInteger x)

instance KnownNat4 i h1 h2 o => Num (Network i h1 h2 o) where
  Net a b c + Net d e f = Net (a + d) (b + e) (c + f)
  Net a b c - Net d e f = Net (a - d) (b - e) (c - f)
  Net a b c * Net d e f = Net (a * d) (b * e) (c * f)
  abs    (Net a b c)    = Net (abs    a) (abs    b) (abs    c)
  signum (Net a b c)    = Net (signum a) (signum b) (signum c)
  negate (Net a b c)    = Net (negate a) (negate b) (negate c)
  fromInteger x         = Net (fromInteger x) (fromInteger x) (fromInteger x)

instance KnownNat2 i o => Fractional (Layer i o) where
  Layer w1 b1 / Layer w2 b2 = Layer (w1 / w2) (b1 / b2)
  recip (Layer w b)         = Layer (recip w) (recip b)
  fromRational x            = Layer (fromRational x) (fromRational x)

instance KnownNat4 i h1 h2 o => Fractional (Network i h1 h2 o) where
  Net a b c / Net d e f = Net (a / d) (b / e) (c / f)
  recip (Net a b c)     = Net (recip a) (recip b) (recip c)
  fromRational x        = Net (fromRational x) (fromRational x) (fromRational x)

instance KnownNat n => MWC.Variate (R n) where
  uniform g = randomVector <$> MWC.uniform g <*> pure Uniform
  uniformR (l, h) g = (\x -> x * (h - l) + l) <$> MWC.uniform g

instance KnownNat2 m n => MWC.Variate (L m n) where
  uniform g = uniformSample <$> MWC.uniform g <*> pure 0 <*> pure 1
  uniformR (l, h) g = (\x -> x * (h - l) + l) <$> MWC.uniform g

instance KnownNat2 i o => MWC.Variate (Layer i o) where
  uniform g = Layer <$> MWC.uniform g <*> MWC.uniform g
  uniformR (l, h) g = (\x -> x * (h - l) + l) <$> MWC.uniform g

instance KnownNat4 i h1 h2 o => MWC.Variate (Network i h1 h2 o) where
  uniform g = Net <$> MWC.uniform g <*> MWC.uniform g <*> MWC.uniform g
  uniformR (l, h) g = (\x -> x * (h - l) + l) <$> MWC.uniform g


-- ========================================================================= --
-- Basic math functions with back propagation

matVec :: forall m n . (KnownNat m, KnownNat n) => Op '[ L m n, R n ] (R m)
matVec = op2' $ \m v -> (forward m v, backward m v)
  where
    forward :: L m n -> R n -> R m
    forward m v = m #> v

    backward :: L m n -> R n -> Maybe (R m) -> (L m n, R n)
    backward m v (fromMaybe 1 -> g) = (g `outer` v, tr m #> g)


dot :: forall n . KnownNat n => Op '[ R n, R n ] Double
dot = op2' $ \x y -> (forward x y, backward x y)
  where
    forward :: R n -> R n -> Double
    forward x y = x <.> y

    backward :: R n -> R n -> Maybe Double -> (R n, R n)
    backward x y = \case
      Nothing -> (y, x)
      Just g  -> (konst g * y, x * konst g)


scale :: forall n . KnownNat n => Op '[ Double, R n ] (R n)
scale = op2' $ \a x -> (forward a x, backward a x)
  where
    forward :: Double -> R n -> R n
    forward a x = konst a * x

    backward :: Double -> R n -> Maybe (R n) -> (Double, R n)
    backward a x = \case
      Nothing -> (HM.sumElements (extract x      ), konst a    )
      Just g  -> (HM.sumElements (extract (x * g)), konst a * g)


vsum :: forall n . KnownNat n => Op '[ R n ] Double
vsum = op1' $ \x -> (forward x, backward)
  where
    forward :: R n -> Double
    forward = HM.sumElements . extract

    backward :: Maybe Double -> R n
    backward = maybe 1 konst


logistic :: Floating a => a -> a
logistic x = 1 / (1 + exp (-x))


-- ========================================================================= --
-- run backpropagation
type LayerCtx i o = '[R i, Layer i o]
type NetCtx i h1 h2 o = '[ R i, Network i h1 h2 o ]

runLayer :: forall i o s . (KnownNat i, KnownNat o) => BPOp s '[ R i, Layer i o ] (R o)
runLayer = withInps $ \(decombinate -> (x, l)) -> do
  w :< b :< _ <- partsVar gTuple l
  y <- opVar matVec (w :< x :< nil)
  return $ y + b
  where
    decombinate
      :: Prod (BVar s (LayerCtx i o)) '[R i, Layer i o]
      -> (BVar s (LayerCtx i o) (R i), BVar s (LayerCtx i o) (Layer i o))
    decombinate (x :< l :< _) = (x, l)


runNetwork :: KnownNat4 i h1 h2 o => BPOp s (NetCtx i h1 h2 o) (R o)
runNetwork = withInps $ \(x :< n :< _) -> do
  l1 :< l2 :< l3 :< _ <- partsVar gTuple n

  y <- bindVar $ liftB (bpOp runLayer) (x :< l1 :< nil)
  -- or
  z <- bindVar $ (bpOp runLayer) .$ (logistic y :< l2 :< nil)
  -- or
  r <- (bpOp runLayer) ~$ (logistic z :< l3 :< nil)

  bpOp softmax ~$ only r


softmax :: KnownNat n => BPOp s '[ R n ] (R n)
softmax = withInps $ \(x :< _) -> do
  expX <- bindVar (exp x)
  totX <- vsum ~$ (expX :< nil)
  scale        ~$ (1 / totX :< expX :< nil)


crossEntropy :: forall s n . KnownNat n => R n -> BPOpI s '[ R n ] Double
crossEntropy targ (r :< _) = negate (dot .$ (log r :< only t))
  where
    t :: BVar s '[R n] (R n)
    t = constVar targ


softMaxCrossEntropy :: forall s n . KnownNat n => R n -> BPOpI s '[ R n ] Double
softMaxCrossEntropy targ (r :< Ø) =
  realToFrac tsum * log (vsum .$ (r :< Ø)) - (dot .$ (r :< t :< Ø))
  where
    tsum :: Double
    tsum = HM.sumElements . extract $ targ

    t :: BVar s '[R n] (R n)
    t = constVar targ


trainStep
    :: forall i h1 h2 o. KnownNat4 i h1 h2 o
    => Double
    -> R i
    -> R o
    -> Network i h1 h2 o
    -> Network i h1 h2 o
trainStep r !x !t !n =
  case grad of
    (_ :< I gN :< _) -> n - (realToFrac r * gN)
    otherwise        -> error "impossible"

  where
    grad :: Tuple '[R i, Network i h1 h2 o]
    grad = gradBPOp (netErr t) (x ::< n ::< Ø)


netErr
    :: (KnownNat i, KnownNat h1, KnownNat h2, KnownNat o)
    => R o
    -> BPOp s '[ R i, Network i h1 h2 o ] Double
netErr t = do
    y <- runNetwork
    implicitly (crossEntropy t) -$ (y :< Ø)


main :: IO ()
main = MWC.withSystemRandom $ \g -> do
    Just test  <- loadMNIST "data/t10k-images-idx3-ubyte"  "data/t10k-labels-idx1-ubyte"
    putStrLn "Loaded data."
    net0 <- MWC.uniformR @(Network 784 300 100 9) (-0.5, 0.5) g
    createDirectoryIfMissing True "bench-results"
    t <- getZonedTime
    let test0   = head test
        tstr    = formatTime defaultTimeLocale "%Y%m%d-%H%M%S" t
    defaultMainWith defaultConfig
          { reportFile = Just $ "bench-results/mnist-bench_" ++ tstr ++ ".html"
          , timeLimit  = 10
          } [
        bgroup "gradient" [
          let testBP     x y = getI . index (IS IZ) $
                  gradBPOp (netErr y) (x ::< net0 ::< Ø)
            in  bench "bp"     $ nf (uncurry testBP) test0
          ]
      , bgroup "descent" [
          let testBP     x y = trainStep 0.02 x y net0
            in  bench "bp"     $ nf (uncurry testBP) test0
          ]
      , bgroup "run" [
          let testBP     x   = evalBPOp runNetwork (x ::< net0 ::< Ø)
          in  bench "bp"     $ nf testBP (fst test0)
          ]
      ]

loadMNIST
    :: FilePath
    -> FilePath
    -> IO (Maybe [(R 784, R 9)])
loadMNIST fpI fpL = runMaybeT $ do
    i <- MaybeT          $ decodeIDXFile       fpI
    l <- MaybeT          $ decodeIDXLabelsFile fpL
    d <- MaybeT . return $ labeledIntData l i
    r <- MaybeT . return $ for d (bitraverse mkImage mkLabel . swap)
    -- liftIO . evaluate $ force r
    liftIO . undefined $ force r
  where
    mkImage :: VU.Vector Int -> Maybe (R 784)
    mkImage = create . VG.convert . VG.map (\i -> fromIntegral i / 255)
    mkLabel :: Int -> Maybe (R 9)
    mkLabel n = create $ HM.build 9 (\i -> if round i == n then 1 else 0)
