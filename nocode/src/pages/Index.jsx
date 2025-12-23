import { CardContent, CardHeader, Card, CardTitle } from '@/components/ui/card';
import { Textarea } from '@/components/ui/textarea';
import { SelectItem, Select, SelectContent, SelectValue, SelectTrigger } from '@/components/ui/select';
import { toast } from 'sonner';
import { useRef, useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Loader2, Send, Upload } from 'lucide-react';
import { Button } from '@/components/ui/button';
const Index = () => {
  const [selectedModel, setSelectedModel] = useState('qwen-vl');
  const [question, setQuestion] = useState('');
  const [image, setImage] = useState(null);
  const [isThinking, setIsThinking] = useState(false);
  const [response, setResponse] = useState(null);
  const [streamedText, setStreamedText] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const responseEndRef = useRef(null);

  // 模拟流式输出
  const simulateStream = (text, callback) => {
    let index = 0;
    const interval = setInterval(() => {
      if (index < text.length) {
        callback(text.substring(0, index + 1));
        index++;
      } else {
        clearInterval(interval);
      }
    }, 30); // 控制流式输出的速度
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = () => {
        setImage(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSubmit = async () => {
    if (!image || !question.trim()) {
      toast.error('请上传图像并输入问题');
      return;
    }

    setIsThinking(true);
    setResponse(null);
    setStreamedText('');
    setIsStreaming(true);

    try {
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // 根据选择的模型生成不同的假数据响应
      let mockResponse;
      if (selectedModel === 'patho-r1' || selectedModel === 'qwen-vl-thinking') {
        // Reason类型模型响应
        mockResponse = {
          thinking: '我正在分析您上传的病理图像。根据图像特征，我观察到细胞形态异常，核质比增大，核分裂象增多。这些特征提示可能存在恶性肿瘤。',
          answer: '根据病理图像分析，该样本显示细胞异型性明显，核质比失调，核分裂活跃。结合临床信息，建议进一步进行免疫组化检查以明确诊断。初步考虑为高级别上皮内瘤变或早期浸润癌可能。'
        };
      } else {
        // Chat类型模型响应
        mockResponse = {
          answer: '根据病理图像分析，该样本显示细胞异型性明显，核质比失调，核分裂活跃。结合临床信息，建议进一步进行免疫组化检查以明确诊断。初步考虑为高级别上皮内瘤变或早期浸润癌可能。'
        };
      }
      
      // 模拟流式输出
      if (mockResponse.thinking) {
        simulateStream(mockResponse.thinking, (text) => {
          setStreamedText(text);
        });
      }
      
      // 等待思考过程流式输出完成
      setTimeout(() => {
        simulateStream(mockResponse.answer, (text) => {
          setStreamedText(prev => prev + text);
        });
      }, mockResponse.thinking ? 2000 : 0);
      
      // 等待流式输出完成
      setTimeout(() => {
        setResponse(mockResponse);
        setIsStreaming(false);
      }, (mockResponse.thinking ? 4000 : 2000));
    } catch (error) {
      toast.error('分析失败，请重试');
      setIsStreaming(false);
    } finally {
      setIsThinking(false);
    }
  };

  // 滚动到最新消息
  useEffect(() => {
    responseEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [streamedText]);

  return (
    <div className="container mx-auto py-8 px-4">
      <h1 className="text-3xl font-bold text-center mb-8">病理视觉语言模型分析系统</h1>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <Card>
          <CardHeader>
            <CardTitle>上传病理图像</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
              {image ? (
                <div className="space-y-4">
                  <img 
                    src={image} 
                    alt="上传的病理图像" 
                    className="mx-auto object-cover max-h-64 rounded-lg" 
                  />
                  <Button 
                    variant="outline" 
                    onClick={() => setImage(null)}
                  >
                    重新上传
                  </Button>
                </div>
              ) : (
                <div className="space-y-4">
                  <Upload className="mx-auto h-12 w-12 text-gray-400" />
                  <div>
                    <label htmlFor="image-upload" className="cursor-pointer">
                      <span className="text-blue-600 hover:text-blue-800">点击上传</span> 或拖拽图像到此处
                    </label>
                    <input
                      id="image-upload"
                      type="file"
                      accept="image/*"
                      className="hidden"
                      onChange={handleImageUpload}
                    />
                  </div>
                  <p className="text-sm text-gray-500">支持 JPG, PNG, GIF 格式</p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>模型配置</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-2">
              <label className="text-sm font-medium">选择模型</label>
              <Select value={selectedModel} onValueChange={setSelectedModel}>
                <SelectTrigger>
                  <SelectValue placeholder="选择模型" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="qwen-vl">Qwen-VL (Chat)</SelectItem>
                  <SelectItem value="qwen-vl-v1">Qwen-VL v1</SelectItem>
                  <SelectItem value="qwen-vl-v2">Qwen-VL v2</SelectItem>
                  <SelectItem value="lingshu">Lingshu</SelectItem>
                  <SelectItem value="patho-r1">Patho-R1 (Reason)</SelectItem>
                  <SelectItem value="qwen-vl-thinking">Qwen-VL (Thinking)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">输入问题</label>
              <Textarea
                placeholder="请描述您想要分析的病理特征或提出问题..."
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                rows={4}
              />
            </div>

            <Button 
              className="w-full" 
              onClick={handleSubmit}
              disabled={isThinking || !image || !question.trim()}
            >
              {isThinking ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  分析中...
                </>
              ) : (
                <>
                  <Send className="mr-2 h-4 w-4" />
                  开始分析
                </>
              )}
            </Button>
          </CardContent>
        </Card>
      </div>

      {(response || isStreaming) && (
        <Card className="mt-8">
          <CardHeader>
            <CardTitle>分析结果</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {isStreaming && (
              <div className="bg-blue-50 p-4 rounded-lg">
                <h3 className="font-medium text-blue-800 mb-2">实时输出</h3>
                <div className="text-sm text-blue-700 whitespace-pre-wrap">
                  {streamedText}
                  <span className="animate-pulse">|</span>
                </div>
                <div ref={responseEndRef} />
              </div>
            )}
            {response && (
              <>
                {response.thinking && (
                  <div className="bg-blue-50 p-4 rounded-lg">
                    <h3 className="font-medium text-blue-800 mb-2">思考过程</h3>
                    <div className="text-sm text-blue-700 whitespace-pre-wrap">
                      {response.thinking}
                    </div>
                  </div>
                )}
                <div className="bg-green-50 p-4 rounded-lg">
                  <h3 className="font-medium text-green-800 mb-2">分析结果</h3>
                  <div className="text-sm text-green-700 whitespace-pre-wrap">
                    {response.answer}
                  </div>
                </div>
              </>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default Index;
