const PaymentProcessor = require('./payment-processor');

describe('PaymentProcessor', () => {
  let processor;
  let mockApiClient;

  beforeEach(() => {
    mockApiClient = {
      post: jest.fn(),
      get: jest.fn()
    };
    processor = new PaymentProcessor(mockApiClient);
  });

  describe('processPayment', () => {
    it('should process payment successfully', async () => {
      const mockResponse = {
        data: {
          id: 'txn_123',
          status: 'completed'
        }
      };
      mockApiClient.post.mockResolvedValue(mockResponse);

      const result = await processor.processPayment(100, 'USD', 'card');

      expect(result).toEqual({
        success: true,
        transactionId: 'txn_123',
        status: 'completed'
      });
      expect(mockApiClient.post).toHaveBeenCalledWith('/payments', {
        amount: 100,
        currency: 'USD',
        paymentMethod: 'card'
      });
    });

    it('should handle payment processing errors', async () => {
      mockApiClient.post.mockRejectedValue(new Error('Network error'));

      const result = await processor.processPayment(100, 'USD', 'card');

      expect(result).toEqual({
        success: false,
        error: 'Network error'
      });
    });

    it('should validate amount is positive', async () => {
      await expect(processor.processPayment(0, 'USD', 'card'))
        .rejects.toThrow('Invalid amount');
    });

    it('should validate currency code', async () => {
      await expect(processor.processPayment(100, 'US', 'card'))
        .rejects.toThrow('Invalid currency code');
    });

    it('should require payment method', async () => {
      await expect(processor.processPayment(100, 'USD', null))
        .rejects.toThrow('Payment method required');
    });
  });

  describe('refundPayment', () => {
    it('should process refund successfully', async () => {
      const mockResponse = {
        data: {
          id: 'refund_123',
          status: 'completed'
        }
      };
      mockApiClient.post.mockResolvedValue(mockResponse);

      const result = await processor.refundPayment('txn_123', 50);

      expect(result).toEqual({
        success: true,
        refundId: 'refund_123',
        status: 'completed'
      });
      expect(mockApiClient.post).toHaveBeenCalledWith('/payments/txn_123/refund', {
        amount: 50
      });
    });

    it('should handle refund errors', async () => {
      mockApiClient.post.mockRejectedValue(new Error('Refund failed'));

      const result = await processor.refundPayment('txn_123', 50);

      expect(result).toEqual({
        success: false,
        error: 'Refund failed'
      });
    });

    it('should require transaction ID', async () => {
      await expect(processor.refundPayment(null, 50))
        .rejects.toThrow('Transaction ID required');
    });

    it('should validate refund amount', async () => {
      await expect(processor.refundPayment('txn_123', 0))
        .rejects.toThrow('Invalid refund amount');
    });
  });

  describe('getPaymentStatus', () => {
    it('should get payment status successfully', async () => {
      const mockResponse = {
        data: {
          status: 'completed',
          amount: 100,
          currency: 'USD'
        }
      };
      mockApiClient.get.mockResolvedValue(mockResponse);

      const result = await processor.getPaymentStatus('txn_123');

      expect(result).toEqual({
        status: 'completed',
        amount: 100,
        currency: 'USD'
      });
      expect(mockApiClient.get).toHaveBeenCalledWith('/payments/txn_123');
    });

    it('should require transaction ID', async () => {
      await expect(processor.getPaymentStatus(null))
        .rejects.toThrow('Transaction ID required');
    });

    it('should handle API errors', async () => {
      mockApiClient.get.mockRejectedValue(new Error('Payment not found'));

      await expect(processor.getPaymentStatus('txn_123'))
        .rejects.toThrow('Failed to get payment status: Payment not found');
    });
  });
});
